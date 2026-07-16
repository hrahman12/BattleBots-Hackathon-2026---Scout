"""Data pipeline: scrape bot pages through Bright Data and fill bots.json.

Scope: reboot era only (2015-2023 / ABC + Discovery seasons). BattleBots also
ran a Comedy Central era (1999-2002), but that data is thinner online and less
consistently structured per fight — reboot-era bots have the richest,
best-documented fight histories, so that's where this dataset stays.

This is the sponsor-integration layer. It routes fetches through Bright Data's
Web Unlocker API so the project genuinely uses their infrastructure.

Setup (on your laptop, not inside Claude):
    pip install requests beautifulsoup4
    export BRIGHTDATA_API_TOKEN="your-account-api-token"   # brightdata.com/cp/setting/users
    python scraper.py

Zone note: if you've run Bright Data's MCP once, it auto-creates an
'mcp_unlocker' zone. Otherwise create a Web Unlocker zone in the dashboard
and set BRIGHTDATA_ZONE to its name.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

API_URL = "https://api.brightdata.com/request"
TOKEN = os.environ.get("BRIGHTDATA_API_TOKEN")
ZONE = os.environ.get("BRIGHTDATA_ZONE", "mcp_unlocker")
DATA_PATH = Path(__file__).parent / "bots.json"

# Roster-driven scraping: instead of a hand-maintained TARGETS dict, loop over
# reboot_roster.txt (one candidate bot name per line) and derive each wiki URL.
ROSTER_PATH = Path(__file__).parent / "reboot_roster.txt"
WIKI_BASE = "https://battlebots.fandom.com/wiki/"
REQUEST_DELAY_S = 1.0  # be polite to the wiki / Bright Data between fetches

# reboot_roster.txt was auto-scraped from wiki season-competitor lists and is
# noisy: alongside real bot names it picked up generic weapon/format words
# ("drum", "flipper", "heavyweight", "wildcard") and non-bot entries
# ("YouTube", an award, a tournament path). Scraping those wastes Bright Data
# credits and pollutes bots.json, so skip them by default. Every key here is a
# confirmed NON-bot; set skip_non_bots=False in load_roster() to scrape anyway.
NON_BOT_TERMS = {
    "another robot", "circular saw", "clamping arms", "cutting saw", "disk",
    "disks", "drisk", "drone", "drum", "drum spinner", "flamethrower",
    "flipper", "hammer saw", "hammer saws", "heavyweight", "killsaws",
    "lifting arms", "middleweight", "minibot", "paddle", "paddles",
    "pulverizer", "robot", "rumble", "screws", "self-right",
    "self-righting mechanism", "self-righting panel", "spinning disk",
    "vertical spinner", "wildcard", "wildcards", "youtube",
    "most destructive robot",  # an award, not a bot
    "road to the giant nut",   # tournament path, not a bot
}


def bot_url(name: str) -> str:
    """Map a bot name to its BattleBots Fandom wiki URL (spaces -> underscores,
    everything else percent-encoded so unicode / punctuation names resolve)."""
    slug = name.strip().replace(" ", "_")
    return WIKI_BASE + quote(slug, safe="_")


def load_roster(skip_non_bots: bool = True) -> list[str]:
    """Read candidate names, drop blanks and case-insensitive dupes, and (by
    default) filter the generic non-bot terms the source list is polluted with."""
    seen: set[str] = set()
    names: list[str] = []
    skipped: list[str] = []
    for line in ROSTER_PATH.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        if skip_non_bots and key in NON_BOT_TERMS:
            skipped.append(name)
            continue
        names.append(name)
    if skipped:
        print(f"Skipped {len(skipped)} non-bot terms: {', '.join(skipped)}\n")
    return names


def fetch(url: str) -> str:
    """Fetch a page through Bright Data Web Unlocker (raw HTML)."""
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"zone": ZONE, "url": url, "format": "raw"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.text


FAILURE_KEYWORDS = re.compile(
    r"(counted out|lost drive|stopped moving|no longer (?:spinning|moving)|"
    r"weapon (?:stopped|died|failed|disabled|non-functional)|caught fire|"
    r"burst into flames|battery (?:ruptured|punctured|died)|could not self-right|"
    r"unable to self-right|lost a (?:belt|wheel)|high-centered|sheared|"
    r"stuck (?:on|against|in)|smoke)", re.IGNORECASE)

IMMOBILIZATION_HINTS = re.compile(r"counted out|immobil|drive|wheel|self-right", re.IGNORECASE)
HARDWARE_HINTS = re.compile(r"battery|weapon (stopped|died|failed)|belt|fire|smoke|sheared", re.IGNORECASE)

WEAPON_PATTERNS = [
    (re.compile(r"vertical spinn", re.IGNORECASE), "vertical_spinner"),
    (re.compile(r"horizontal spinn|bar spinn", re.IGNORECASE), "horizontal_spinner"),
    (re.compile(r"drum spinn", re.IGNORECASE), "horizontal_spinner"),
    (re.compile(r"flipper|four-bar", re.IGNORECASE), "flipper"),
    (re.compile(r"grappl|hydraulic jaws", re.IGNORECASE), "grappler_flipper"),
    (re.compile(r"hammer\s?saw", re.IGNORECASE), "hammer_saw"),
    (re.compile(r"crush", re.IGNORECASE), "crusher"),
]


def _clean(raw_text: str) -> str:
    """Undo double-escaping artifacts, then strip image/link markup and long
    base64 data URIs so keyword scanning doesn't false-positive on alt text
    or binary noise instead of actual fight-history prose."""
    text = raw_text.replace('\\"', '"').replace("\\n", "\n").replace("\\'", "'")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # markdown images
    text = re.sub(r"data:image/[^)\s]+", " ", text)             # inline data URIs
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)        # markdown links -> just the label text
    return text


def _extract_weapon_type(text: str) -> str:
    for pattern, label in WEAPON_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


def _extract_record(text: str) -> tuple[int | None, int | None]:
    m = re.search(r"Competitive Wins/Losses.{0,60}?Wins:\s*(\d+).{0,40}?Losses:\s*(\d+)", text, re.DOTALL)
    if not m:
        m = re.search(r"[*\s]*Wins:\s*(\d+)\s*[\n*]+\s*Losses:\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_incidents(text: str, bot_name: str, max_incidents: int = 5) -> list[dict]:
    """Sentence-scoped extraction around failure keywords, filtered to only
    keep matches where the bot's OWN name appears in the same sentence
    window. Without this check, a scan easily attributes an opponent's
    defeat to the bot being profiled (e.g. text about 'X counted out
    UltraViolent' reads as an UltraViolent-failure match even when profiling
    X) — which produces exactly backwards exploit advice. Still automated
    and noisier than hand-curated dossiers; spot-check before a real demo.
    """
    incidents = []
    name_first_word = bot_name.split()[0]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for idx, sentence in enumerate(sentences):
        if not FAILURE_KEYWORDS.search(sentence):
            continue
        if name_first_word.lower() not in sentence.lower():
            continue  # keyword sentence doesn't mention this bot by name — skip rather than guess
        window = " ".join(sentences[max(0, idx - 1):idx + 2])
        snippet = " ".join(window.split())[:250]
        opp_m = re.search(r"vs\.?\s+([A-Z][A-Za-z0-9!'\- ]{2,25})", " ".join(sentences[max(0, idx - 6):idx]))
        event = f"vs. {opp_m.group(1).strip()}" if opp_m else "documented incident"
        incidents.append({"event": event, "failure": snippet})
        if len(incidents) >= max_incidents:
            break
    return incidents


def _classify_trigger(incidents: list[dict]) -> str | None:
    if not incidents:
        return None
    text_blob = " ".join(i["failure"] for i in incidents)
    immob = len(IMMOBILIZATION_HINTS.findall(text_blob))
    hw = len(HARDWARE_HINTS.findall(text_blob))
    if hw >= immob and hw > 0:
        return "late_fight_hardware"
    if immob > 0:
        return "immobilization"
    return None


def parse_bot_page(html: str, name: str) -> dict:
    """Extract a dossier automatically from a fandom bot page's markdown text.

    Fully automated — no manual reading. This trades precision for scale:
    expect noisier incident snippets and occasional misclassified weapon
    types compared to hand-verified dossiers. Good enough to seed a
    roster-wide defect pattern; worth spot-checking before a real demo.
    """
    text = _clean(html)
    weapon_type = _extract_weapon_type(text)
    wins, losses = _extract_record(text)
    incidents = _extract_incidents(text, name)
    trigger = _classify_trigger(incidents)

    if trigger == "immobilization":
        summary = "Automated scan flags a recurring immobilization/count-out pattern in this bot's fight history."
    elif trigger == "late_fight_hardware":
        summary = "Automated scan flags recurring weapon/battery hardware failures in this bot's fight history."
    else:
        summary = "No recurring failure pattern detected by automated scan (may just mean incidents weren't phrased in a way the keyword scan catches)."

    return {
        "name": name,
        "weapon_type": weapon_type,
        "career_wins": wins,
        "career_losses": losses,
        "self_failure_incidents": incidents,
        "defect_summary": summary,
        "exploit": None,
        "failure_trigger": trigger,
        "data_completeness": "auto-scraped — unverified, spot-check before relying on it",
    }


def main() -> None:
    if not TOKEN:
        sys.exit("Set BRIGHTDATA_API_TOKEN first (Bright Data > settings > users).")

    db = json.loads(DATA_PATH.read_text())
    existing = {b["name"].lower(): b for b in db["bots"]}
    roster = load_roster()
    print(f"Scraping {len(roster)} bots via Bright Data (zone: {ZONE})...\n")

    flagged: list[str] = []   # parser returned no weapon_type and/or no incidents
    failed: list[str] = []    # fetch or parse raised
    added = updated = 0

    for i, name in enumerate(roster, 1):
        try:
            print(f"[{i}/{len(roster)}] {name} ...", flush=True)
            html = fetch(bot_url(name))
            record = parse_bot_page(html, name)
        except Exception as e:  # keep the batch alive; one bad page shouldn't kill 180
            print(f"    ! failed: {e}")
            failed.append(f"{name} ({type(e).__name__}: {e})")
            continue

        issues = []
        if record["weapon_type"] == "unknown":
            issues.append("no weapon_type")
        if not record["self_failure_incidents"]:
            issues.append("no incidents")
        if issues:
            flagged.append(f"{name}: {', '.join(issues)}")

        key = name.lower()
        if key in existing:
            existing[key].update(record)
            updated += 1
        else:
            existing[key] = record
            added += 1
        time.sleep(REQUEST_DELAY_S)

    db["bots"] = list(existing.values())
    DATA_PATH.write_text(json.dumps(db, indent=2))

    print(f"\nDone. {added} added, {updated} updated, {len(failed)} failed. "
          f"bots.json now holds {len(db['bots'])} bots.")
    if flagged:
        print(f"\n⚠ SPOT-CHECK these {len(flagged)} — parser returned incomplete data "
              f"(noisier than hand-verified entries):")
        for f in flagged:
            print(f"  - {f}")
    if failed:
        print(f"\n✗ FETCH/PARSE FAILURES ({len(failed)}) — likely no wiki page at the guessed URL:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
