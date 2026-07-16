# battlebots_mcp — Combat Robotics Scout MCP Server

An MCP server that gives any AI agent scouting superpowers over combat
robotics data: bot dossiers, **key-defect analysis** ("predict the defect
before the win"), matchup prediction with a quantified defect-impact number,
and an attack-by-attack battle plan.

**Scope: reboot era (2015-2023) only.** Roughly 150-220 distinct competitors
across seven World Championships. Reboot bots have the densest, best-
documented fight histories and are what most fans and judges recognize.

Data pipeline runs through **Bright Data** (see `scraper.py`).

## Roster status

- **10 bots loaded**, in two tiers:
  - **5 hand-verified** (Tombstone, End Game, Bite Force, Witch Doctor,
    Minotaur) — read and cross-checked by hand, high confidence.
  - **5 auto-scraped** (HyperShock, DeathRoll, Copperhead, Yeti, Whiplash) —
    extracted by `scraper.py`'s automated parser, no manual reading. Faster,
    noisier: incident snippets are keyword-matched sentences, not narrative
    summaries, and weapon-type detection is a simple pattern match.
- **`reboot_roster.txt`** has ~220 candidate names extracted from the wiki's
  own season-competitor templates — this is the target list for scraping
  the *entire* reboot era. Point `scraper.py`'s `TARGETS` at it and run the
  full batch with your own Bright Data key; it's built to run unattended.

### A real bug this caught, worth knowing about

The first version of the auto-parser matched failure keywords anywhere near
the bot's name, which meant it sometimes attributed an **opponent's** loss to
the bot being profiled (e.g. "HyperShock counted out UltraViolent" got read
as evidence of a HyperShock weakness). Fixed by requiring the bot's name
appear in the *same sentence* as the failure keyword. Any further roster
expansion should spot-check a handful of bots after scraping — automated
extraction at this scale will have edge cases like this.

## Quickstart

```bash
pip install mcp requests
python server.py                                    # run (stdio)
npx @modelcontextprotocol/inspector python server.py  # poke tools in a UI
```

## Connect to Claude

**Claude Code (fastest for the demo):**
```bash
claude mcp add battlebots -- python /full/path/to/server.py
```

**Claude Desktop** — add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "battlebots": {
      "command": "python",
      "args": ["/full/path/to/server.py"]
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `battlebots_list_bots` | List database, filter by weapon type |
| `battlebots_get_bot` | Full dossier for one bot |
| `battlebots_get_defects` | Failure-mode analysis + exploit strategy |
| `battlebots_predict_matchup` | Win probability + defect-impact breakdown |
| `battlebots_battle_plan` | Attack-by-attack script exploiting a defender's defect |

## Running the full reboot-era scrape

```bash
pip install requests
export BRIGHTDATA_API_TOKEN="your-token"
python scraper.py    # currently targets a small TARGETS dict — expand from reboot_roster.txt
```

`parse_bot_page()` in `scraper.py` is a real, working extractor now (weapon
type, win/loss record, name-checked failure incidents, trigger
classification) — not a stub. Scaling to all ~220 names is a config change
(swap `TARGETS` for the full roster list) plus patience: at ~1 credit per
page it's well inside the 5,000/month free tier, the real cost is just
runtime and eyeballing a sample afterward.

## 9 PM demo script

1. Connect the server to Claude live. Ask: *"Scout Tombstone for me —
   what's his weakness?"* → Claude calls `battlebots_get_defects`, returns
   the recoil/battery defect pattern and "survive the first minute" exploit.
2. *"Who wins, Tombstone or End Game, and how much does his weakness
   actually matter?"* → prediction plus the defect-impact percentage-point
   breakdown, honestly flagging missing data.
3. The finale: *"Build me a fight strategy to beat Tombstone."* → Claude
   chains get_bot → get_defects → predict_matchup → battle_plan **on its
   own** and outputs a full game plan. That unscripted composition is the
   MCP payoff: you built tools, the agent built the workflow.

## Honest-pitch notes (judges like these)

- Matchup weights are labeled priors, not trained values. The upgrade path
  is real: scrape fight records → logistic regression → replace
  `FEATURE_WEIGHTS`. Say this out loud; it reads as rigor.
- "Defects" = statistically documented self-failure modes from public
  fight records, not mechanical inspection.
- The auto-scraped tier is honestly labeled as such (`data_completeness`
  field) and is lower-confidence than the hand-verified tier — the tool
  never hides which bots got which treatment.
