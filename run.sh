#!/usr/bin/env bash
set -e

runs=${1:-1}

for dataset in weibo twitter acm aps dblp; do
    if [ ! -f "prepared/$dataset.pt" ]; then
        python -m ccgl.prepare --dataset "$dataset" \
            --raw "datasets/$dataset/dataset.txt" --output "prepared/$dataset.pt"
    fi
    python -m ccgl.train --data "prepared/$dataset.pt" \
        --output "results/main/$dataset" --runs "$runs"
done
