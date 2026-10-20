#!/usr/bin/env python3
"""Merge independent MLC_SATIN shard statistics without torch.distributed."""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_shards", type=int, required=True)
    args = p.parse_args()
    paths = [os.path.join(args.stats_dir, f"{args.dataset}.shard{i}of{args.num_shards}.json") for i in range(args.num_shards)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing shard statistics:\n" + "\n".join(missing))
    rows = [json.load(open(path)) for path in paths]
    stats = [[sum(row["stats"][i][j] for row in rows) for j in range(len(rows[0]["stats"][i]))] for i in range(4)]
    exact = sum(row["exact"] for row in rows)
    total = rows[0]["total"]
    tp = sum(stats[0])
    tn = sum(stats[1])
    acc = (tp + tn) / (sum(sum(row) for row in stats) + 1e-15)
    print(f"{args.dataset} (SATIN train split; samples={total}):")
    print(f"  Accuracy (micro over labels): {acc * 100:.2f}%")
    print(f"  Exact-match sample accuracy: {exact / total * 100:.2f}%")
    print(f"  Shards merged: {args.num_shards}")


if __name__ == "__main__":
    main()
