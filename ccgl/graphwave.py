"""Batched GraphWave with the reference Chebyshev expansion (order 30)."""

import numpy as np
import torch


@torch.no_grad()
def embed(cascades, observation, device, batch_size=128):
    features, lengths = [], []
    order = 30
    x = torch.cos((2 * torch.arange(1, order + 1, device=device,
                                   dtype=torch.float64) - 1) * torch.pi / (2 * order))
    basis = [torch.ones_like(x), x]
    for _ in range(order - 1):
        basis.append(2 * x * basis[-1] - basis[-2])
    basis = torch.stack(basis)
    time_points = torch.linspace(0, 100, 16, device=device, dtype=torch.float64)

    for start in range(0, len(cascades), batch_size):
        batch = [c.events[:100] for c in cascades[start:start + batch_size]]
        node_lists = [list(dict.fromkeys(node for path, _ in events for node in path[-2:]))
                      for events in batch]
        sizes = torch.tensor([len(nodes) for nodes in node_lists], device=device)
        n = int(sizes.max())
        adjacency = np.zeros((len(batch), n, n), dtype=np.float64)
        indices, times = [], []
        for i, (events, nodes) in enumerate(zip(batch, node_lists)):
            index = {node: j for j, node in enumerate(nodes)}
            indices.append([index[path[-1]] for path, _ in events])
            times.append([1 if len(path) == 1 else 1 - t / observation for path, t in events])
            for path, t in events:
                if len(path) > 1:
                    u, v = index[path[-1]], index[path[-2]]
                    adjacency[i, u, v] = adjacency[i, v, u] = 1 - t / observation
        a = torch.as_tensor(adjacency, device=device)
        degree = a.sum(-1)
        inv = torch.where(degree > 1e-10, degree.rsqrt(), 0)
        # L - I = -D^(-1/2) A D^(-1/2).
        operator = -inv[:, :, None] * a * inv[:, None, :]
        taus = -torch.log(torch.tensor([0.95, 0.80], device=device,
                                      dtype=torch.float64))[None] * (sizes[:, None].double() / 2).sqrt()
        kernel = torch.exp(-taus[:, :, None] * (x + 1))
        coefficients = 2 / order * (kernel @ basis.T)
        coefficients[:, :, 0] /= 2
        previous = torch.eye(n, device=device, dtype=torch.float64).expand(len(batch), n, n)
        current = operator
        heat = coefficients[:, :, 0, None, None] * previous[:, None]
        heat += coefficients[:, :, 1, None, None] * current[:, None]
        for k in range(2, order + 1):
            following = 2 * operator.bmm(current) - previous
            heat += coefficients[:, :, k, None, None] * following[:, None]
            previous, current = current, following
        heat = torch.where(heat > (1e-4 / sizes)[:, None, None, None], heat, 0)
        angles = heat[..., None] * time_points
        row_mask = (torch.arange(n, device=device)[None] < sizes[:, None])[:, None, :, None, None]
        real = (angles.cos() * row_mask).sum(2) / sizes[:, None, None, None]
        imaginary = (angles.sin() * row_mask).sum(2) / sizes[:, None, None, None]
        signatures = torch.stack((real, imaginary), -1).permute(0, 2, 1, 3, 4)
        signatures = signatures.reshape(len(batch), n, 64).float()
        for i, (index, time) in enumerate(zip(indices, times)):
            embedding = signatures[i, index].clone()
            embedding[:, 0] = torch.tensor(time, device=device)
            features.append(embedding.cpu())
            lengths.append(len(time))
        if (start // batch_size) % 20 == 0:
            print(f"  embedded {min(start + batch_size, len(cascades))}/{len(cascades)}", flush=True)

    return {"features": torch.cat(features), "lengths": torch.tensor(lengths),
            "ids": [c.id for c in cascades],
            "labels": torch.tensor([c.label for c in cascades], dtype=torch.float32)}
