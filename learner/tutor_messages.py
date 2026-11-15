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


def _value(o):
    return o["sev"] * 0.95 ** (o["dist"] / 2.0)          # discounted value for the learner (speed 2 m/s)


def hint(concept, s):
    """Hints only point the learner to victims no blue robot is heading for (the robots are usually closer and
    would get there first), and that the learner can still reach in time."""
    o = s["options"]
    free = {j: v for j, v in o.items() if not v["claimed"] and v["slack"] >= 0 and
            v.get("my_eta", 0) < v.get("robot_eta", 1e9)}            # nobody going, and the learner gets there first
    best = max(free.values(), key=_value) if free else None
    if concept == "priority":
        j, v = _most(free, "dist", lambda x: x["sev"] == 3)
        if v:
            return (f"There's a critical victim {v['dist']:.0f} m from you - you're the closest. "
                    f"Severity counts for more than distance.")
        crit_ours = [v for v in o.values() if v["sev"] == 3 and (v["claimed"] or v.get("robot_eta", 1e9) <= v.get("my_eta", 0))]
        if crit_ours and best:
            return (f"We'll get to the critical victim{'s' if len(crit_ours) > 1 else ''} before you can. "
                    f"The most valuable one left for you: the {SEV[best['sev']]} victim {best['dist']:.0f} m away.")
        if best:
            return f"Look at severity first: the {SEV[best['sev']]} victim {best['dist']:.0f} m away is worth the most to you."
        return "Look at the severity rings before you pick: red (critical) is worth the most."
    if concept == "deadline":
        timed = {j: v for j, v in o.items() if not v["claimed"] and v["time_left"] is not None and
                 v.get("my_eta", 0) < v.get("robot_eta", 1e9)}
        j, v = _most(timed, "slack")
        if v:
            reach = v["dist"] / 2.0
            if v["slack"] < 0:
                return (f"The {SEV[v['sev']]} victim {v['dist']:.0f} m away has only {v['time_left']:.0f} s left - "
                        f"you'd need about {reach:.0f} s. Too far for you: pick someone you can still reach.")
            return (f"The {SEV[v['sev']]} victim {v['dist']:.0f} m away has {v['time_left']:.0f} s left and you're the "
                    f"closest - you can make it in about {reach:.0f} s if you go now.")
        return "Check the timers: can you reach that victim before their time runs out?"
    if concept == "coordination":
        j, v = _most(o, "mate_dist", lambda x: x["claimed"])
        if v:
            tip = f" - for example the {SEV[best['sev']]} victim {best['dist']:.0f} m from you" if best else ""
            return (f"I'm already heading to the {SEV[v['sev']]} victim {v['dist']:.0f} m from you. "
                    f"Pick someone else so we don't both go{tip}.")
        return "Watch our dashed lines - they show where we're going."
    if concept == "trade_off":
        if free and best:
            near = min(free.values(), key=lambda x: x["dist"])
            if near is not best:
                return (f"Closest for you: the {SEV[near['sev']]} victim {near['dist']:.0f} m away "
                        f"({near['sev']} x 0.95^{near['dist'] / 2:.0f}s = {_value(near):.2f}). The "
                        f"{SEV[best['sev']]} one {best['dist']:.0f} m away is worth {best['sev']} x "
                        f"0.95^{best['dist'] / 2:.0f}s = {_value(best):.2f}. Travel time matters.")
        return ("A close victim or a far, more severe one? Count the travel time: value = severity x 0.95^(seconds "
                "to reach).")
    if concept == "adaptation":
        j, v = _most(o, "sev", lambda x: x["new"], reverse=True)          # the most severe new victim
        if v and v["claimed"]:
            return (f"New report: a {SEV[v['sev']]} victim, {v['dist']:.0f} m away - one of us is already going. "
                    f"Is your own target still the best one?")
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
