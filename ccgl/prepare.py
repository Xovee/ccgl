"""Prepare two AugSIM views and GraphWave features from Parquet cascades."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import Cascade, DATASETS, read_cascades
from .graphwave import embed


def graph_degrees(events):
    edges = {tuple(sorted(path[-2:])) for path, _ in events if len(path) > 1}
    degree = Counter(node for edge in edges for node in edge)
    return degree


def reaction_times(cascade):
    times = [t for _, t in cascade.events[:100]]
    return np.diff(times)


def augment(cascades, observation, rng, strength=0.1, theta=0.1):
    intervals = np.concatenate([reaction_times(c) for c in cascades])
    location = float(intervals.min())
    scale = float(intervals.mean() - location)
    views = []
    for cascade in cascades:
        events = list(cascade.events[:100])
        degree = graph_degrees(events)
        total_degree = sum(degree.values())
        local_time = float(reaction_times(cascade).mean())
        expanded = list(events)
        next_node = min(0, *(node for path, _ in events for node in path)) - 1
        # Every observed node, including the root, may attract one new adopter.
        for path, time in events:
            probability = strength * len(events) * degree[path[-1]] / total_degree
            if rng.random() < probability:
                delay = theta * local_time + (1 - theta) * (location + rng.exponential(scale))
                new_time = int(time + delay)
                if new_time < observation:
                    expanded.append((path + (next_node,), new_time))
                    next_node -= 1
        expanded_degree = graph_degrees(expanded)
        root = events[0][0][-1]
        leaves = {path[-1]: path[-2] for path, _ in expanded
                  if len(path) > 1 and path[-1] != root and expanded_degree[path[-1]] == 1}
        normalizer = sum(expanded_degree[parent] for parent in leaves.values())
        removed = {node for node, parent in leaves.items()
                   if rng.random() < strength * len(leaves) * expanded_degree[parent] / normalizer}
        kept = [event for event in expanded if event[0][-1] not in removed]
        kept.sort(key=lambda event: event[1])
        views.append(Cascade(cascade.id, kept[:100], 0))
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="A cascade Parquet file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = json.loads(pq.ParquetFile(args.raw).schema_arrow.metadata[b"cascade_metadata"])
    if source["dataset"] != args.dataset:
        parser.error("--dataset must match the dataset recorded in the Parquet file.")
    torch.set_num_threads(4)
    splits = read_cascades(args.raw, args.dataset)
    labeled_ids = {c.id for name in ("train", "val", "test") for c in splits[name]}
    overlap = sum(c.id in labeled_ids for c in splits["unlabeled"])
    splits["unlabeled"] = [c for c in splits["unlabeled"] if c.id not in labeled_ids]
    with args.raw.open("rb") as raw:
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    manifest = {
        "dataset": args.dataset, "raw_sha256": digest, "source": source,
        "split_seed": DATASETS[args.dataset][3], "augmentation_seed": args.seed,
        "counts": {name: len(cascades) for name, cascades in splits.items()},
        "overlap_candidates": overlap, "include_overlap": False,
        "torch": str(torch.__version__), "augmentation": "AugSIM, ratio=0.1, reaction intervals, theta=0.1",
    }
    print(json.dumps(manifest), flush=True)
    rng = np.random.default_rng(args.seed)
    observation = DATASETS[args.dataset][0]
    splits["view1"] = augment(splits["train"], observation, rng)
    splits["view2"] = augment(splits["train"], observation, rng)
    splits["unlabeled1"] = augment(splits["unlabeled"], observation, rng)
    splits["unlabeled2"] = augment(splits.pop("unlabeled"), observation, rng)
    result = {"manifest": manifest}
    for name, cascades in splits.items():
        print(f"Embedding {args.dataset}/{name}: {len(cascades)} cascades", flush=True)
        result[name] = embed(cascades, observation, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
