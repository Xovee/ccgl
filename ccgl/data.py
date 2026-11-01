"""Raw cascade parsing and the original deterministic labeled-data splits."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
import random

import torch


# observation horizon, prediction horizon, minimum observed size, split seed
DATASETS = {
    "weibo": (3600, 86400, 10, "xovee"),
    "twitter": (172800, 2764800, 10, 0),
    "acm": (3, 10, 10, "xovee"),
    "aps": (1095, 7305, 10, 0),
    "dblp": (5, 20, 5, "xovee"),
}


@dataclass
class Cascade:
    id: str
    events: list  # (root-to-node path, adoption time), sorted by adoption time
    label: int


def read_cascades(path, dataset):
    observation, prediction, minimum, seed = DATASETS[dataset]
    labeled, unlabeled = [], []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            identifier, _, published, _, adoptions = line.rstrip().split("\t")
            events = []
            future_size = 0
            for adoption in adoptions.split():
                nodes, time = adoption.rsplit(":", 1)
                time = int(time)
                future_size += time < prediction
                if time < observation:
                    events.append((tuple(map(int, nodes.split("/"))), time))
            if len(events) < minimum:
                continue
            events.sort(key=lambda event: event[1])
            if dataset in ("weibo", "twitter"):
                # UTC+8 reproduces the published Twitter counts; host TZ is irrelevant.
                date = datetime.fromtimestamp(float(published), timezone.utc) + timedelta(hours=8)
                if dataset == "weibo":
                    is_labeled = 8 <= date.hour < 18
                    is_unlabeled = 18 <= date.hour < 23
                else:
                    is_labeled = not (date.month == 4 and date.day > 10)
                    is_unlabeled = date.month != 3 and not (date.month == 4 and date.day < 10)
            else:
                cutoff = {"acm": "2006", "aps": "1997-12", "dblp": "1997"}[dataset]
                is_labeled, is_unlabeled = published <= cutoff, published > cutoff
            cascade = Cascade(identifier, events, future_size - len(events))
            if is_labeled:
                labeled.append(cascade)
            if is_unlabeled:
                unlabeled.append(cascade)

    ids = [cascade.id for cascade in labeled]
    random.Random(seed).shuffle(ids)
    end_train, end_val = ceil(len(ids) * 0.5), ceil(len(ids) * 0.6)
    memberships = [set(ids[:end_train]), set(ids[end_train:end_val]), set(ids[end_val:])]
    splits = {name: [c for c in labeled if c.id in members]
              for name, members in zip(("train", "val", "test"), memberships)}
    # The historical pipeline shuffled the training rows a second time before
    # selecting every 10th/100th row for the 10%/1% labeled subsets.
    random.Random(seed).shuffle(splits["train"])
    splits["unlabeled"] = unlabeled
    return splits


def padded(split, device, max_length=100):
    """Pad once when loading a split, rather than copying/padding every batch."""
    lengths = split["lengths"].long()
    rows = torch.arange(len(lengths)).repeat_interleave(lengths)
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
    columns = torch.arange(len(split["features"])) - offsets.repeat_interleave(lengths)
    x = torch.zeros(len(lengths), max_length, split["features"].shape[-1])
    x[rows, columns] = split["features"]
    return x.to(device)
