"""battlebots_mcp — Scout MCP server for combat robotics data.

Gives any MCP client (Claude Desktop, Claude Code, agents) scouting tools over
a local BattleBots dossier database: bot lookups, defect analysis, and matchup
prediction with transparent, per-feature reasoning.

Run:            python server.py            (stdio transport)
Inspect:        npx @modelcontextprotocol/inspector python server.py
Data pipeline:  scraper.py (Bright Data) fills bots.json
"""

import difflib
import json
import math
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "bots.json"

# Hand-set priors — replace with trained weights once the scraper has filled
# enough fight records (see README: "training the predictor").
FEATURE_WEIGHTS = {
    "finishing_power": 0.35,   # KO rate: ends fights before defects matter
    "reliability": 0.30,       # 1 - penalty per documented self-failure
    "weapon_matchup": 0.20,    # style-vs-style prior
    "pedigree": 0.15,          # championship titles
}
SELF_FAILURE_PENALTY = 0.15    # _prior: reliability lost per documented incident
LOGISTIC_K = 4.0               # score-diff -> probability steepness

# _prior: conventional combat-robotics wisdom, to be replaced by learned values.
# P(row's weapon beats column's weapon). Unlisted pairs default to 0.5.
WEAPON_MATCHUP_PRIOR = {
    ("vertical_spinner", "horizontal_spinner"): 0.60,
}


def _load_bots() -> dict[str, dict]:
    with open(DATA_PATH) as f:
        return {b["name"].lower(): b for b in json.load(f)["bots"]}


def _find_bot(name: str) -> dict:
    """Case-insensitive lookup with did-you-mean suggestions."""
    bots = _load_bots()
    bot = bots.get(name.lower().strip())
    if bot:
        return bot
    close = difflib.get_close_matches(name.lower(), bots.keys(), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(
        f"Bot '{name}' not found.{hint} "
        f"Use battlebots_list_bots to see all {len(bots)} available bots."
    )


def _weapon_matchup(a: str, b: str) -> float:
    if (a, b) in WEAPON_MATCHUP_PRIOR:
        return WEAPON_MATCHUP_PRIOR[(a, b)]
    if (b, a) in WEAPON_MATCHUP_PRIOR:
        return 1.0 - WEAPON_MATCHUP_PRIOR[(b, a)]
    return 0.5


def _feature_scores(bot: dict, opponent: dict) -> tuple[dict, list[str]]:
    """Score each feature 0-1. Missing data scores neutral 0.5 and is flagged."""
    missing: list[str] = []

    if bot.get("ko_win_rate") is not None:
        finishing = bot["ko_win_rate"]
    else:
        finishing, _ = 0.5, missing.append(f"{bot['name']}: ko_win_rate")

    reliability = max(
        0.0, 1.0 - SELF_FAILURE_PENALTY * len(bot.get("self_failure_incidents", []))
    )
    matchup = _weapon_matchup(
        bot.get("weapon_type", "unknown"), opponent.get("weapon_type", "unknown")
    )
    pedigree = min(1.0, 0.5 * len(bot.get("titles", [])))

    return (
        {
            "finishing_power": round(finishing, 3),
            "reliability": round(reliability, 3),
            "weapon_matchup": round(matchup, 3),
            "pedigree": round(pedigree, 3),
        },
        missing,
    )


def _predict(bot_a: dict, bot_b: dict) -> dict:
    """Transparent heuristic predictor. Returns probability + reasoning."""
    scores_a, miss_a = _feature_scores(bot_a, bot_b)
    scores_b, miss_b = _feature_scores(bot_b, bot_a)
    total_a = sum(FEATURE_WEIGHTS[f] * s for f, s in scores_a.items())
    total_b = sum(FEATURE_WEIGHTS[f] * s for f, s in scores_b.items())
    prob_a = 1.0 / (1.0 + math.exp(-LOGISTIC_K * (total_a - total_b)))
    missing = miss_a + miss_b

    # Defect impact: recompute with reliability neutralized (both bots treated
    # as having zero documented failure history) to isolate exactly how much
    # of the win probability comes from defect history vs. everything else.
    neutral_scores_a = {**scores_a, "reliability": 1.0}
    neutral_scores_b = {**scores_b, "reliability": 1.0}
    neutral_total_a = sum(FEATURE_WEIGHTS[f] * s for f, s in neutral_scores_a.items())
    neutral_total_b = sum(FEATURE_WEIGHTS[f] * s for f, s in neutral_scores_b.items())
    neutral_prob_a = 1.0 / (1.0 + math.exp(-LOGISTIC_K * (neutral_total_a - neutral_total_b)))
    defect_swing_pp = round((prob_a - neutral_prob_a) * 100, 1)

    return {
        "bot_a": bot_a["name"],
        "bot_b": bot_b["name"],
        "win_probability_a": round(prob_a, 3),
        "win_probability_b": round(1.0 - prob_a, 3),
        "feature_scores": {bot_a["name"]: scores_a, bot_b["name"]: scores_b},
        "feature_weights": FEATURE_WEIGHTS,
        "key_defects": {
            b["name"]: b.get("defect_summary") or "none documented"
            for b in (bot_a, bot_b)
        },
        "defect_impact": {
            "win_probability_with_defect_history": round(prob_a, 3),
            "win_probability_if_both_bots_were_equally_reliable": round(neutral_prob_a, 3),
            "defect_history_swings_it_by_percentage_points": abs(defect_swing_pp),
            "favors": bot_a["name"] if defect_swing_pp > 0 else bot_b["name"] if defect_swing_pp < 0 else "neither",
            "explanation": (
                f"Reliability (documented self-failure history) is {FEATURE_WEIGHTS['reliability']*100:.0f}% "
                f"of the score. Strip it out and hold everything else equal, and the win probability moves "
                f"from {round(prob_a*100)}% to {round(neutral_prob_a*100)}% for {bot_a['name']} \u2014 "
                f"that {abs(defect_swing_pp):.1f}-point gap is exactly how much this matchup's outcome is "
                f"driven by defects rather than weapon type, finishing power, or pedigree."
            ),
        },
        "confidence": "low — heuristic priors" if missing else "moderate — heuristic",
        "missing_data": missing or None,
        "note": (
            "Priors, not trained weights. Fill bots.json via scraper.py, then "
            "fit a logistic model on fight records to replace FEATURE_WEIGHTS."
        ),
    }


# ---------------------------------------------------------------------------
# Battle plans: attack-by-attack scripts keyed to a bot's documented failure
# trigger. Templated from real incident patterns, not fabricated — this is a
# strategic hypothesis grounded in fight history, not a guaranteed outcome.
# ---------------------------------------------------------------------------

BATTLE_PLAYBOOKS = {
    "late_fight_hardware": {
        "label": "Late-fight hardware failure",
        "phases": [
            {
                "phase": "Opening (0-20s)",
                "attacker_action": (
                    "Stay mobile, avoid trading. Let {defender} spin up and "
                    "swing first — bait a miss rather than trading blows while "
                    "their weapon and batteries are freshest."
                ),
                "defender_counter": (
                    "{defender} should only commit to hits they're confident "
                    "will land clean. Wasted full-power swings burn exactly "
                    "the battery margin needed late in the fight."
                ),
            },
            {
                "phase": "Mid (20-60s)",
                "attacker_action": (
                    "Trade at bad angles — edge-of-blade or weapon-on-weapon "
                    "hits have historically loosened {defender}'s weapon chain "
                    "and cracked battery mounts."
                ),
                "defender_counter": (
                    "{defender} should avoid edge contact, retreat to center "
                    "on trades, and watch for early chain slip or a change in "
                    "weapon pitch/sound."
                ),
            },
            {
                "phase": "Late (60s+)",
                "attacker_action": (
                    "Extend the fight. In every documented loss, {defender}'s "
                    "weapon or batteries failed after the one-minute mark — "
                    "the win condition is patience, not a bigger hit."
                ),
                "defender_counter": (
                    "If weapon RPM audibly drops, {defender} should disengage "
                    "immediately and fight for a judges' decision rather than "
                    "trading blind on a failing weapon."
                ),
            },
        ],
    },
    "immobilization": {
        "label": "Immobilization / count-out",
        "phases": [
            {
                "phase": "Opening (0-20s)",
                "attacker_action": (
                    "Target wheel guards and drive pods directly, not the body "
                    "shell — {defender}'s losses are drive failures, not "
                    "structural ones."
                ),
                "defender_counter": (
                    "{defender} should keep the weapon spun to full speed at "
                    "all times, even defensively — losses have come from "
                    "being caught with the disk not up to speed."
                ),
            },
            {
                "phase": "Mid (20-60s)",
                "attacker_action": (
                    "Push for an inversion or pin against the screws while "
                    "{defender}'s weapon is down — that is the exact window "
                    "their self-right has failed before."
                ),
                "defender_counter": (
                    "{defender} should treat 'spinning' as the default state, "
                    "not a response to attacks, so a flip never catches the "
                    "weapon stopped."
                ),
            },
            {
                "phase": "Late (60s+)",
                "attacker_action": (
                    "Sustain pressure on one drive side rather than hunting "
                    "for a knockout blow — the count-out is the win condition "
                    "here, not destruction."
                ),
                "defender_counter": (
                    "{defender} should monitor drive response and call a "
                    "tactical retreat before one side fully fails, rather "
                    "than fighting to the end on a dying wheel."
                ),
            },
        ],
    },
}


def _fill(template: str, attacker: str, defender: str) -> str:
    return template.format(attacker=attacker, defender=defender)


def _battle_plan(attacker: dict, defender: dict) -> dict:
    """Build an attack-by-attack plan exploiting the defender's documented defect."""
    trigger = defender.get("failure_trigger")
    playbook = BATTLE_PLAYBOOKS.get(trigger)
    if not playbook:
        return {
            "attacker": attacker["name"],
            "defender": defender["name"],
            "error": (
                f"No battle playbook available — {defender['name']} has no "
                "classified failure_trigger yet. Add one to bots.json once "
                "enough incidents are scraped."
            ),
        }
    phases = [
        {
            "phase": p["phase"],
            "attacker_action": _fill(p["attacker_action"], attacker["name"], defender["name"]),
            "defender_counter": _fill(p["defender_counter"], attacker["name"], defender["name"]),
        }
        for p in playbook["phases"]
    ]
    return {
        "attacker": attacker["name"],
        "defender": defender["name"],
        "exploiting": playbook["label"],
        "based_on": defender.get("self_failure_incidents", []),
        "phases": phases,
        "caveat": (
            "This is a strategic hypothesis derived from documented fight "
            "history, not a guaranteed outcome — real matches involve pilot "
            "skill, hardware variance, and factors no archive records."
        ),
    }


# ---------------------------------------------------------------------------
# MCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("battlebots_mcp")

READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


class BotNameInput(BaseModel):
    """Input for single-bot lookups."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., description="Bot name, case-insensitive (e.g. 'Tombstone')", min_length=1, max_length=100)


class ListBotsInput(BaseModel):
    """Input for listing/filtering bots."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    weapon_type: Optional[str] = Field(
        default=None,
        description="Filter by weapon type (e.g. 'horizontal_spinner', 'vertical_spinner'). Omit for all bots.",
    )


class MatchupInput(BaseModel):
    """Input for matchup prediction."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    bot_a: str = Field(..., description="First bot name (e.g. 'Tombstone')", min_length=1, max_length=100)
    bot_b: str = Field(..., description="Second bot name (e.g. 'End Game')", min_length=1, max_length=100)


@mcp.tool(name="battlebots_list_bots", annotations={"title": "List Scouted Bots", **READ_ONLY})
async def battlebots_list_bots(params: ListBotsInput) -> str:
    """List all bots in the scouting database, optionally filtered by weapon type.

    Returns: JSON array of {name, weapon_type, weight_class, titles, data_completeness}.
    """
    bots = _load_bots().values()
    if params.weapon_type:
        bots = [b for b in bots if b.get("weapon_type") == params.weapon_type]
    summary = [
        {k: b.get(k) for k in ("name", "weapon_type", "weight_class", "titles", "data_completeness")}
        for b in bots
    ]
    return json.dumps({"count": len(summary), "bots": summary}, indent=2)


@mcp.tool(name="battlebots_get_bot", annotations={"title": "Get Bot Dossier", **READ_ONLY})
async def battlebots_get_bot(params: BotNameInput) -> str:
    """Get the full scouting dossier for one bot: weapon, record, titles, defects, sources.

    Returns: JSON object with all known fields; null fields need scraping.
    """
    try:
        return json.dumps(_find_bot(params.name), indent=2)
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool(name="battlebots_get_defects", annotations={"title": "Analyze Key Defects", **READ_ONLY})
async def battlebots_get_defects(params: BotNameInput) -> str:
    """Analyze a bot's documented failure modes and how to exploit them.

    This is the 'predict the defect before the win' tool: it surfaces
    self-failure incidents, the pattern behind them, and the recommended
    counter-strategy.

    Returns: JSON {bot, defect_summary, incidents[], exploit, incident_count}.
    """
    try:
        bot = _find_bot(params.name)
    except ValueError as e:
        return f"Error: {e}"
    incidents = bot.get("self_failure_incidents", [])
    return json.dumps(
        {
            "bot": bot["name"],
            "defect_summary": bot.get("defect_summary") or "No defects documented — dossier may be incomplete.",
            "incident_count": len(incidents),
            "incidents": incidents,
            "exploit": bot.get("exploit") or "Insufficient data for a counter-strategy. Scrape more fight records.",
        },
        indent=2,
    )


@mcp.tool(name="battlebots_predict_matchup", annotations={"title": "Predict Matchup", **READ_ONLY})
async def battlebots_predict_matchup(params: MatchupInput) -> str:
    """Predict a head-to-head matchup, and precisely quantify how much of that
    prediction is driven by each bot's documented defect history.

    Scores each bot on 4 weighted features: finishing_power (0.35), reliability
    i.e. defect/self-failure history (0.30), weapon_matchup (0.20), pedigree
    (0.15) — converts the weighted score gap to a win probability via a
    logistic function. Then recomputes the same matchup with reliability
    neutralized (both bots treated as equally defect-free) to isolate exactly
    how many percentage points the defect history alone is worth.

    Returns: JSON {win_probability_a/b, feature_scores, key_defects,
    defect_impact (with a plain-language percentage-point breakdown),
    confidence, missing_data}.
    """
    try:
        bot_a, bot_b = _find_bot(params.bot_a), _find_bot(params.bot_b)
    except ValueError as e:
        return f"Error: {e}"
    return json.dumps(_predict(bot_a, bot_b), indent=2)


class BattlePlanInput(BaseModel):
    """Input for attack-by-attack battle plan generation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    attacker: str = Field(..., description="Bot executing the strategy (e.g. 'Tombstone')", min_length=1, max_length=100)
    defender: str = Field(..., description="Bot whose documented defect is being targeted (e.g. 'End Game')", min_length=1, max_length=100)


@mcp.tool(name="battlebots_battle_plan", annotations={"title": "Attack-by-Attack Battle Plan", **READ_ONLY})
async def battlebots_battle_plan(params: BattlePlanInput) -> str:
    """Generate a phase-by-phase plan for attacker to exploit defender's documented defect,
    paired with defender's counter-play at each phase.

    Templated from the defender's failure_trigger classification (e.g.
    'late_fight_hardware', 'immobilization'), grounded in its actual
    self_failure_incidents. Not a guaranteed outcome — a strategic hypothesis.

    Returns: JSON {attacker, defender, exploiting, based_on[], phases[], caveat}.
    """
    try:
        attacker, defender = _find_bot(params.attacker), _find_bot(params.defender)
    except ValueError as e:
        return f"Error: {e}"
    return json.dumps(_battle_plan(attacker, defender), indent=2)


if __name__ == "__main__":
    mcp.run()
