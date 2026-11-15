"""What the blue robots say when the pedagogical agent decides to help.

message(action, situation, chosen=None, results=None, notes=None) -> (speaker, text)
Messages are filled in from the live situation (distances, time left, severities), so they refer to what the
learner can see on screen, and each one links the situation to the RL idea behind it.
"""
from learner.tutor_env import ASK0, CONCEPTS, EXPLAIN0, HINT0, QUIET

SEV = {1: "minor", 2: "moderate", 3: "critical"}
RL_IDEA = {
    "priority": "Reward design: rescuing a victim is worth 10 x severity, so a critical victim is worth three "
                "times a minor one.",
    "deadline": "Episodes have time limits: a victim who is lost gives -5 x severity, and going for someone you "
                "cannot reach in time wastes the whole trip.",
    "coordination": "Multi-agent credit assignment: the team shares one reward, so two robots on the same victim "
                    "earn nothing extra - spread out.",
    "trade_off": "Discounting: a reward that arrives later is worth less. Value = severity x 0.95^(seconds to "
                 "reach), so a near minor victim can beat a far critical one - or not.",
    "adaptation": "Non-stationary environments: when new information arrives (a new report), the best plan can "
                  "change - an RL agent re-decides instead of sticking to its old target.",
}


def _most(opts, key, pred=lambda o: True, reverse=False):
    c = [(j, o) for j, o in opts.items() if pred(o)]
    if not c:
        return None, None
    return (max if reverse else min)(c, key=lambda x: x[1][key])


def hint(concept, s):
    o = s["options"]
    if concept == "priority":
        j, v = _most(o, "dist", lambda x: x["sev"] == 3 and not x["claimed"])
        if v:
            return f"There's a critical victim {v['dist']:.0f} m from you. Severity counts for more than distance."
        return "Look at the severity rings before you pick: red (critical) is worth the most."
    if concept == "deadline":
        j, v = _most(o, "slack", lambda x: x["time_left"] is not None)
        if v:
            reach = v["dist"] / 2.0
            return (f"The {SEV[v['sev']]} victim {v['dist']:.0f} m away has {v['time_left']:.0f} s left - you need "
                    f"about {reach:.0f} s to get there. Check who you can still reach in time.")
        return "Check the timers: can you reach that victim before their time runs out?"
    if concept == "coordination":
        j, v = _most(o, "mate_dist", lambda x: x["claimed"])
        if v:
            return (f"I'm already heading to the {SEV[v['sev']]} victim {v['dist']:.0f} m from you. "
                    f"Pick someone else so we don't both go.")
        return "Watch our dashed lines - they show where we're going."
    if concept == "trade_off":
        return ("A close victim or a far, more severe one? Count the travel time: value = severity x 0.95^(seconds "
                "to reach).")
    if concept == "adaptation":
        j, v = _most(o, "dist", lambda x: x["new"])
        if v:
            return f"New report: a {SEV[v['sev']]} victim, {v['dist']:.0f} m away. Does that change your plan?"
        return "A new victim was just reported - is your current target still the best one?"
    return ""


QUESTIONS = {
    "priority": "Before you go: which of these victims is worth the most to the team, and why?",
    "deadline": "Quick check: which victims can you still reach before their time runs out?",
    "coordination": "Which victims are we (the blue robots) already covering? Where are you most useful?",
    "trade_off": "Near and less severe, or far and more severe - which gives more value once you count the "
                 "travel time?",
    "adaptation": "Something new just appeared. Would you keep your plan or change it?",
}


def explain(concept, results, notes):
    ok = results.get(concept)
    idea = RL_IDEA[concept]
    if ok is False:
        note = next((n for n in notes if _about(n, concept)), None) or FALLBACK[concept]
        return f"Not the best choice: {note}. {idea}"
    return f"Good choice. {idea}"


FALLBACK = {
    "priority": "a more severe victim was about the same effort",
    "deadline": "that victim could not be reached before their time ran out",
    "coordination": "a blue robot was already on its way there",
    "trade_off": "counting travel time, another victim was worth more",
    "adaptation": "a victim had just been reported, and it changed what the best choice was",
}


def _about(note, concept):
    keys = {"priority": "same effort", "deadline": "left but", "coordination": "already on its way",
            "trade_off": "counting travel time", "adaptation": "just been reported"}
    return keys[concept] in note


def message(action, situation, results=None, notes=None, speaker=None):
    """Text for a tutor action. speaker: 'R1' or 'R2' (default: alternate by concept)."""
    if action == QUIET:
        return None
    if action >= EXPLAIN0:
        c = CONCEPTS[action - EXPLAIN0]
        text = explain(c, results or {}, notes or [])
    elif action >= ASK0:
        c = CONCEPTS[action - ASK0]
        text = QUESTIONS[c]
    else:
        c = CONCEPTS[action - HINT0]
        text = hint(c, situation)
    return speaker or ("R1" if CONCEPTS.index(c) % 2 == 0 else "R2"), text
