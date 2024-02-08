
from tabulate import tabulate
import pandas as pd
from pathlib import Path

from score import get_scores

outputs = Path("data/outputs")

ref = {
    "test": Path("data/SHROOM_test-labeled"),
    "val": Path("data/SHROOM_dev-v2")
}

tracks = ["aware","agnostic"]
splits = ["val","test"]

if __name__ == "__main__":
    for track in tracks:
        for split in splits:
            files = outputs.glob(f'**/{split}*{track}*.json')

            scores = get_scores(files,ref[split],is_val=split == "val")

            df = pd.DataFrame(scores).T
            latex_table = tabulate(df, tablefmt="latex",floatfmt=".3f")
            with open(Path(__file__).parent / f"tables/{split}-{track}.tex","w") as f:
                f.write(latex_table)