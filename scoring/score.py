# command: python3 $program/score.py $input/res $input/ref $output
# description: run scoring program on any data

import argparse
import collections
import json
from pathlib import Path
from typing import List

from scipy.stats import spearmanr


def get_scores(submitted_files: List[Path], ref_dir: Path, is_val: bool):

    prefix = "val" if is_val else "test"

    scores = collections.defaultdict(dict)
    for sub_file in submitted_files:

        # assert sub_file.name in (f'{prefix}.model-aware.json', f'{prefix}.model-agnostic.json'), \
        #     f'Invalid file name in submission `{sub_file.name}`'

        ref_file = ref_dir / sub_file.name
        with open(ref_file, "r") as istr:
            ref_data = json.load(istr)
            if not is_val:
                ref_data = sorted(ref_data, key=lambda item: item["id"])

        with open(sub_file, "r") as istr:
            sub_data = json.load(istr)
            if not is_val:
                sub_data = sorted(sub_data, key=lambda item: int(item["id"]))

        expected_keys = {"label", "p(Hallucination)"}
        if not is_val:
            expected_keys.add("id")
        assert all(
            (set(sub.keys()) & expected_keys) == expected_keys
            for sub in sub_data
        ), f"Not all datapoints contain all expected keys"

        assert len(sub_data) == len(ref_data), (
            f'Invalid number of items in "{sub_file.name}": '
            f"expected {len(ref_data)}, got {len(sub_data)}"
        )

        if not is_val:
            assert [int(it["id"]) for it in sub_data] == [
                it["id"] for it in ref_data
            ], f'Invalid id in submission file "{sub_file.name}" ({set(int(it["id"]) for it in sub_data) - set(it["id"] for it in ref_data)} )'

        assert all(
            sub["label"] in {"Hallucination", "Not Hallucination"}
            for sub in sub_data
        ), "not all labels have the correct format"

        category = "-".join(sub_file.parts[2:-1]).replace(
            "_", "-"
        )  # ref_file.with_suffix('').name.split('-')[-1]
        scores[category]["acc"] = sum(
            [
                sub["label"] == ref["label"]
                for sub, ref in zip(sub_data, ref_data)
            ]
        ) / len(sub_data)
        scores[category]["rho"] = spearmanr(
            [sub["p(Hallucination)"] for sub in sub_data],
            [ref["p(Hallucination)"] for ref in ref_data],
        )[0]
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=str)
    parser.add_argument("--is_val", action="store_true")

    args = parser.parse_args()

    submitted_files = list(args.submission.glob("**/*.json"))
    n_ref_files = len(list(args.reference.glob("*.json")))
    assert (
        0 < len(submitted_files) <= n_ref_files
    ), f"Expected 0 < n files <= {n_ref_files}, got {len(submitted_files)}"

    scores = get_scores(submitted_files, args.reference, args.is_val)

    with open(args.output, "w") as ostr:
        for cat in scores:
            for fn in "acc", "rho":
                print(f"{cat}_{fn}:{scores[cat][fn]}", file=ostr)
