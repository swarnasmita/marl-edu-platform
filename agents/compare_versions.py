"""Before/after comparison of the MAPPO agents: version 1 vs version 2 (fixes from the behaviour tests).

    python agents/compare_versions.py
Reads results/v1_before_fixes/ (version 1) and results/ (version 2), writes results/before_after.md
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "results", "v1_before_fixes")
NEW = os.path.join(ROOT, "results")


def eval_rows(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "eval_results.csv"))))
    return {(r["teammate"], r["policy"]): r for r in rows}


def md_tables(path):
    """{first cell: [cells]} for every markdown table row in a file."""
    out = {}
    for line in open(path):
        if line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out[cells[1] if cells[0][:1].isdigit() and len(cells) > 3 and "(" in line else cells[0]] = cells
    return out


def pct(cell):
    m = re.match(r"([\d.]+)%", cell)
    return float(m.group(1)) if m else float("nan")


def num(cell):
    m = re.match(r"-?[\d.]+", cell)
    return float(m.group(0)) if m else float("nan")


def main():
    if not os.path.exists(os.path.join(OLD, "eval_results.csv")):
        sys.exit("results/v1_before_fixes/ is missing")
    e_old, e_new = eval_rows(OLD), eval_rows(NEW)
    mappo = "MAPPO (hierarchical)"
    L = ["# MAPPO version 1 vs version 2 (fixes from the behaviour tests)", "",
         "Version 2 changes: (1) victims that can no longer be reached in time are not offered while a feasible one "
         "exists; (2) a small reward penalty (-1) for choosing the victim the other blue robot is already heading for; "
         "(3) a feature telling each blue robot which victim the human appears to be heading for (inferred from "
         "movement; right about 74% of the time). Both versions: seed 0, same training settings, same test scenarios.",
         "", "## Behaviour tests: controlled situations (pass rate)", "",
         "| Situation | Version 1 | Version 2 | Deadline-aware rule |", "|---|---|---|---|"]
    b_old, b_new = md_tables(os.path.join(OLD, "behaviour_report.md")), md_tables(os.path.join(NEW, "behaviour_report.md"))
    for key, cells in b_new.items():
        if len(cells) >= 6 and "(" in cells[3]:
            old = b_old[key][3] if key in b_old else "- (new test)"
            L.append(f"| {key} | {old} | **{cells[3]}** | {cells[5]} |")
    L += ["", "## Behaviour tests: full episodes (greedy simulated human)", "",
          "| Measure | Version 1 | Version 2 | Greedy rule |", "|---|---|---|---|"]
    for key, cells in b_new.items():
        if key.startswith(("2 ", "3 ", "4 ", "5 ", "-")) and key in b_old and len(cells) >= 4 and "(" not in cells[1]:
            L.append(f"| {key} | {b_old[key][1]} | **{cells[1]}** | {cells[3]} |")
    L += ["", "## Evaluation: 100 unseen scenarios per teammate", "",
          "| Teammate | Metric | Version 1 | Version 2 | Best rule |", "|---|---|---|---|---|"]
    metrics = [("score", "Team score", "{:.1f}"), ("sev", "Severity saved", "{:.1%}"),
               ("success", "Success rate", "{:.0%}"), ("crit_lost", "Critical lost", "{:.2f}"),
               ("duplicate_rate", "Duplicate choices", "{:.1%}"), ("wasted_trips", "Wasted trips", "{:.2f}"),
               ("wait_moderate", "Wait: moderate (s)", "{:.1f}")]
    for tm in ("greedy", "noisy", "idle"):
        rules = [e_new[(tm, p)] for p in ("Greedy rule (severity / distance)", "Deadline-aware rule")]
        best = max(rules, key=lambda r: float(r["score"]))
        for k, name, f in metrics:
            o, n = e_old.get((tm, mappo)), e_new.get((tm, mappo))
            if o is None or n is None or k not in o:
                continue
            L.append(f"| {tm} | {name} | {f.format(float(o[k]))} | **{f.format(float(n[k]))}** | "
                     f"{f.format(float(best[k]))} |")
    open(os.path.join(NEW, "before_after.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
