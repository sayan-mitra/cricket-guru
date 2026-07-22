#!/usr/bin/env python3
"""Rules gold — hand-authored questions with Law/clause-cited reference answers.

The references are written independently of the corpus text (not scraped from a
chunk) so grading the rules arm against them isn't circular. Several are
staleness-tagged: the rule genuinely changed, so an old Sports-SE-style answer
would now be wrong — good signal for the staleness story.

DRAFTS — verify each reference against the actual MCC Laws / ICC playing
conditions PDFs (curate in the app, same as narrative).

    python -m cricket_guru.eval.gold_rules
"""
import json

from cricket_guru.config import DATA_DIR

# (question, reference, law, staleness)
ITEMS = [
    ("Is there a free hit after a no-ball?",
     "In white-ball cricket (ODIs and T20Is) the delivery after any no-ball is a free hit, on which the batter can be dismissed only in the ways possible off a no-ball (e.g. run out). Tests have no free hit. It's an ICC playing condition, not in the MCC Laws.",
     "ICC PC 21.20", True),
    ("What are the conditions for an LBW dismissal?",
     "Out LBW if the ball is legal, pitches in line or outside off (not outside leg), strikes the batter in line with the stumps first (or outside off if no genuine shot was offered), and would have gone on to hit the stumps.",
     "Law 36", False),
    ("When is a delivery a front-foot no-ball?",
     "It's a front-foot no-ball unless some part of the bowler's front foot, whether grounded or raised, is behind the popping crease on landing.",
     "Law 21.5", False),
    ("How many fielders can be behind square on the leg side?",
     "No more than two fielders may be behind square on the leg side at the instant of delivery; more than two makes it a no-ball.",
     "Law 28.4", False),
    ("How many balls are in an over?",
     "Six legal deliveries. Wides and no-balls do not count and are re-bowled.",
     "Law 17.1", False),
    ("Can a bowler run out the non-striker who backs up too far?",
     "Yes. Since the 2022 Code, running out the non-striker who has left the crease before the bowler would normally release the ball is a fair run out — no longer classed as unfair play.",
     "Law 38.3", True),
    ("How long does a new batter have to be ready to face the next ball?",
     "Under the MCC Laws the incoming batter must be ready to receive within 3 minutes or be given timed out; ICC playing conditions reduce this to 2 minutes.",
     "Law 40 / ICC PC", True),
    ("By how many runs must a team lead to enforce the follow-on in a five-day Test?",
     "200 runs or more (for a match scheduled for five or more days).",
     "Law 14.1.1", False),
    ("When is a shot worth six runs versus four?",
     "Six if the ball clears the boundary on the full; four if it reaches the boundary having bounced or touches the boundary rope.",
     "Law 19", False),
    ("What is the difference between a bye and a leg bye?",
     "A bye is runs scored when the ball misses both bat and body; a leg bye is runs off the body only when the striker attempted a shot or to avoid the ball.",
     "Law 23", False),
    ("Is 'handled the ball' still a way of getting out?",
     "No. Since the 2017 Code it is no longer a separate dismissal — deliberately using a hand not holding the bat to stop the ball is now 'obstructing the field'.",
     "Law 37", True),
    ("How many unsuccessful reviews does each team get under DRS?",
     "Two unsuccessful player reviews per innings under current ICC playing conditions (increased from one).",
     "ICC PC (DRS)", True),
    ("How many bouncers are allowed per over?",
     "Two fast short-pitched deliveries above shoulder height per over in Tests and ODIs; T20Is historically allowed one, now increased to two.",
     "ICC PC", True),
    ("What are the fielding restrictions in an ODI?",
     "Overs 1–10: at most two fielders outside the 30-yard circle; overs 11–40: at most four; overs 41–50: at most five.",
     "ICC ODI PC", True),
    ("When does the ball become dead?",
     "The ball is dead when it finally settles with the wicket-keeper or bowler, when a boundary is scored, when a batter is dismissed, or in the other cases the Law lists; no runs or dismissals count once it is dead.",
     "Law 20", False),
    ("What is the difference between retired hurt and retired out?",
     "A batter who retires due to injury/illness (retired hurt) may resume the innings; a batter who retires for any other reason without the umpires' consent is retired out and may not resume.",
     "Law 25.4", False),

    # --- Reclassified from the narrative gold (route-by-source = rules) ---
    ("If a miscounted 7th delivery of an over is a no-ball, is there a free hit?",
     "An extra ball bowled because the umpire miscounted still counts as a delivery; a no-ball on it is a no-ball, and in white-ball cricket the next delivery is a free hit if the over continues. Once the umpire calls 'Over', the over is complete.",
     "Law 17.5", False),
    ("What is the maximum number of overs that can be bowled in a day of a Test?",
     "There is no fixed maximum — a Test day is bounded by time, not overs. A minimum of 90 overs is scheduled, with the last-hour provision guaranteeing at least 15 further overs.",
     "Law 12 / ICC PC", False),
    ("How is a wide judged on a reverse or switch hit?",
     "The wide lines are judged on the striker's original stance before the shot; switching to a reverse/switch hit does not swap the off/leg wide markings, so a ball wide of the original stance is still a wide.",
     "Law 22 / ICC PC", False),
    ("Can a batter decline a boundary to retain strike?",
     "No. Once the ball touches or crosses the boundary the boundary is scored automatically and the runs stand; batters cannot refuse it to keep strike.",
     "Law 19", False),
    ("Can two batters be dismissed off a single delivery?",
     "No — only one batter can be dismissed from any single delivery.",
     "Law 31", False),
    ("Is underarm bowling allowed?",
     "Underarm bowling is not permitted unless both captains agree before the match; otherwise the umpire calls it a no-ball.",
     "Law 21.1", True),
    ("If the ball hits the stumps off a no-ball, is the batter bowled?",
     "No. A no-ball is still called and the batter cannot be bowled off it; off a no-ball a batter can be out only run out, obstructing the field, or hit the ball twice.",
     "Law 21", True),
    ("After how many overs can a new ball be taken?",
     "In a match of more than one day the fielding captain may demand a new ball after a set number of overs — 80 overs in Tests under current conditions (85 under the older 2000 code).",
     "Law 4 / ICC PC", True),
    ("What happens if the ball is lost during play?",
     "A fielder may call 'lost ball'; the ball is then dead and the batting side is awarded the runs completed plus the one in progress, or the penalty for any infringement, whichever is greater.",
     "Law 20.5", False),
    ("Is a batter obliged to attempt a run after hitting the ball in the air?",
     "No — batters are never obliged to run and may decline. What is not allowed is deliberately obstructing the field.",
     "Law 37", False),
    ("Can a batter hit the ball after it has passed the stumps?",
     "Yes, a batter may play at the ball after it passes the stumps, provided it is not a second strike made solely to guard the wicket (hit the ball twice).",
     "Law 34", False),
    ("Is a trial ball allowed in a match?",
     "No — a bowler may take a trial run-up with the umpire's knowledge but may not bowl a trial delivery; it falls under fair/unfair play.",
     "Law 41", False),
    ("How is a tied Super Over resolved?",
     "Under current ICC/IPL playing conditions a tied Super Over is resolved by bowling further Super Overs until there is a result; the old boundary-count-back tiebreak was abolished after 2019.",
     "ICC/IPL PC", True),
    ("Can the captain be changed during a match?",
     "Yes — the captain may be changed during the match as long as the replacement is one of the nominated eleven (not a substitute).",
     "Law 1.3", False),
    ("What dismissals are possible off a no-ball, a wide, and a free hit?",
     "Off a no-ball: run out, obstructing the field, hit the ball twice. Off a wide: additionally stumped and hit wicket. On a free hit: the same dismissals as off a no-ball.",
     "Laws 21, 22 / ICC PC", False),
    ("Why is polishing/rubbing the ball not ball tampering?",
     "Polishing is legal provided no artificial substance is used and no time is wasted; rubbing on the ground or applying substances is unfair. (The relevant Law is 41, renumbered from Law 42 in the 2017 Code.)",
     "Law 41", True),
    ("When is a boundary from an overthrow scored, and how many runs?",
     "If an overthrow (or a fielder's act) sends the ball to the boundary, the batting side scores the boundary allowance plus the runs already completed, and the run in progress if the batters had crossed at the instant of the throw.",
     "Law 19.8", False),

    # --- Follow-ups added during review ---
    ("In one-day cricket, does the two-fielders-behind-square-on-the-leg-side limit apply across all 50 overs?",
     "Yes — the cap of two fielders behind square on the leg side is a core Law that applies in every format at all times, and breaching it is a no-ball. ODIs add separate powerplay restrictions on fielders outside the 30-yard circle (overs 1-10: max 2; 11-40: max 4; 41-50: max 5), but those are independent of the behind-square-leg limit, which never changes.",
     "Law 28.4 / ICC ODI PC", False),
    ("Does the follow-on depend on the total a team scores or on the lead?",
     "On the lead (the first-innings margin), not the absolute total. The required lead depends on match length: 200 for five days or more, 150 for three or four days, 100 for two days, 75 for one day. So regardless of the totals, a first-innings lead of at least 200 in a Test allows the follow-on.",
     "Law 14.1", False),
]


def main():
    gold = [{
        "id": f"rules-{i+1}", "qtype": "rules", "staleness": stale,
        "question": q, "reference": ref, "law": law,
    } for i, (q, ref, law, stale) in enumerate(ITEMS)]
    out = DATA_DIR / "gold" / "rules_gold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gold, indent=2))
    print(f"wrote {len(gold)} rules questions "
          f"({sum(g['staleness'] for g in gold)} staleness-tagged) -> {out}")


if __name__ == "__main__":
    main()
