"""
Step 2 - EXPLORATORY DATA ANALYSIS (script version)
===================================================
Runs all 20 EDA questions from the brief + the 4 Step-2 analyses and saves
  outputs/eda/qNN_<slug>.png      one chart per question
  outputs/eda/tables/qNN.csv      supporting table
  outputs/eda/eda_insights.md     the question, the statistically-checked answer, and the table
  outputs/eda/eda_insights.json   same, machine-readable (used by the documentation builder)

The very same functions power the Streamlit "EDA" page (src/eda_questions.py).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import eda_questions as Q  # noqa: E402


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    (C.EDA_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.FEATURED_CSV)
    results, md = [], ["# EDA findings\n"]
    section = None
    for qid, sec, title, question, fn in Q.QUESTIONS:
        res = fn(df)
        png = C.EDA_DIR / f"q{qid:02d}_{slug(title)}.png"
        res["fig"].savefig(png, dpi=130, bbox_inches="tight")
        plt.close(res["fig"])
        res["table"].to_csv(C.EDA_DIR / "tables" / f"q{qid:02d}.csv")
        if sec != section:
            md.append(f"\n## {sec}\n")
            section = sec
        md.append(f"### Q{qid}. {question}\n\n![]({png.name})\n\n**Finding:** {res['insight']}\n\n"
                  f"{res['table'].head(12).to_markdown()}\n")
        results.append(dict(id=qid, section=sec, title=title, question=question,
                            insight=res["insight"], image=png.name))
        print(f"Q{qid:>2} done  {title}")
    (C.EDA_DIR / "eda_insights.md").write_text("\n".join(md), encoding="utf-8")
    (C.EDA_DIR / "eda_insights.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} analyses to {C.EDA_DIR}")


if __name__ == "__main__":
    main()
