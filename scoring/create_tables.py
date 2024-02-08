import pandas as pd
from pathlib import Path

from score import get_scores

outputs = Path("data/outputs")

ref = {
    "test": Path("data/SHROOM_test-labeled"),
    "val": Path("data/SHROOM_dev-v2"),
}

tracks = ["aware", "agnostic"]
splits = ["val", "test"]

if __name__ == "__main__":
    for track in tracks:
        for split in splits:
            files = outputs.glob(f"**/{split}*{track}*.json")

            scores = get_scores(files, ref[split], is_val=split == "val")

            scores = dict(sorted(scores.items(), key=lambda item: (item[1]["acc"],item[1]["rho"]),reverse=True))

            df = pd.DataFrame(scores).T
            df.style.set_table_styles(
                [
                    {"selector": "toprule", "props": ":hline;"},
                    {"selector": "midrule", "props": ":hline;"},
                    {"selector": "bottomrule", "props": ":hline;"},
                ],
                overwrite=False,
            ).format({("Numeric", "Floats"): "{:.3f}"}).format_index(
                escape="latex", axis=0
            ).highlight_max(
                axis=None,
                props='textit:--rwrap; textbf:--rwrap;'
            ).to_latex(
                Path(__file__).parent / f"tables/{split}-{track}.tex",
                clines="all;data",
                caption=f"{split.capitalize()}set model-{track}",
                label=f"{split}-{track}",
                column_format="|p{0.4\linewidth}|r|r|",
            )
