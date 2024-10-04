from pathlib import Path

import pandas as pd
from numpy import random

import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph


def save_graph_to_DG_data(g: Data, filename: str):
    """
    Saves a PyTorch Geometric Data object to a `.csv`-file in `DG_data/`.

    Args:
        g (Data): PyTorch Geometric Data object.
        filename (str): Name of the file.
    """
    df = pd.DataFrame(g.edge_index.numpy().T, columns=["source", "destination"])
    df["timestamp"] = g.time.numpy()
    df["state_label"] = 0
    df["w"] = 0.0
    path = Path("DG_data") / filename
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"DG_data/{filename}/{filename}.csv", index=False)


def create_periodic(
    n: int = 250, p: float = 0.03, r: int = 50, poisson_lambda: int = 1
) -> Data:
    """
    Creates a periodic graph using the Erdos-Renyi model with G(n, p).
    Time differences between consecutive edges are sampled using the Poisson
    distribution parameterized with `poisson_lambda`. The resulting graph is
    repeated `r` times to create the final periodic graph.

    Args:
        n (int): Number of nodes.
        p (float): Probability of an edge between two nodes.
        r (int): Number of repetitions.
        poisson_lambda (int): Poisson distribution parameter.

    Returns:
        Data: PyTorch Geometric Data object representing the periodic graph
            with timestamps saved in the `time` attribute.
    """
    edge_index = erdos_renyi_graph(n, p)
    timestamps = torch.cumsum(
        torch.poisson(torch.ones(edge_index.shape[1]) * poisson_lambda), dim=0
    ).int()
    periodic_edge_index = edge_index
    periodic_timestamps = timestamps
    for _ in range(r - 1):
        periodic_edge_index = torch.cat([periodic_edge_index, edge_index], dim=1)
        periodic_timestamps = torch.cat(
            [periodic_timestamps, timestamps + periodic_timestamps[-1]], dim=0
        )
    return Data(edge_index=periodic_edge_index, time=periodic_timestamps)


def create_triadic(
    n: int = 250, n_triangles: int = 10000, poisson_lambda: int = 5
) -> Data:
    """
    Creates a triadic graph by randomly sampling a triplet of nodes and
    connecting them all in the next 3 consecutive time steps. Poisson distribution
    is used to sample the pause time between one triangle and the next.

    Args:
        n (int): Number of nodes.
        n_triangles (int): Number of triangles to sample.
        poisson_lambda (int): Poisson distribution parameter.
    """
    triplets = torch.randint(n, (n_triangles, 3))
    cum_pause_times = torch.cumsum(
        torch.poisson(torch.ones(n_triangles) * poisson_lambda), dim=0
    ).int()

    edge_index_1 = torch.stack([triplets[:, 0], triplets[:, 1]])
    t_1 = cum_pause_times
    edge_index_2 = torch.stack([triplets[:, 1], triplets[:, 2]])
    t_2 = cum_pause_times + 1
    edge_index_3 = torch.stack([triplets[:, 2], triplets[:, 0]])
    t_3 = cum_pause_times + 2

    edge_index = torch.stack([edge_index_1, edge_index_2, edge_index_3], dim=-1).reshape(2, -1)
    t = torch.stack([t_1, t_2, t_3], dim=-1).reshape(-1)
    sorted_idx = torch.argsort(t)

    return Data(edge_index=edge_index[:, sorted_idx], time=t[sorted_idx])


def create_bipartite(
    growth_steps: int = 1,
    r_1: int = 5,
    r_2: int = 5,
    n_cluster: int = 25,
    p_init: float = 0.2,
    p_inc: float = 0.3,
) -> Data:
    """
    Creates the bipartite graph with trend pattern as proposed in our work.

    Args:
        growth_steps (int): Number of growth steps.
        r_1 (int): Number of clusters in the first block.
        r_2 (int): Number of clusters in the second block.
        n_cluster (int): Number of nodes in each cluster.
        p_init (float): Initial probability of an edge between two nodes.
        p_inc (float): Increase in the probability of an edge.
    """

    edge_index = torch.empty((2, 0), dtype=torch.long)
    t = torch.empty((0,), dtype=torch.long)
    block_sizes = [n_cluster] * (r_1 + r_2)
    curr_t = 0
    for i in range(r_1):
        for j in range(r_2):
            edge_probs = torch.zeros((r_1 + r_2), (r_1 + r_2))
            edge_probs[i, j + r_1] = p_init
            for _ in range(growth_steps + 1):
                curr_edge_index = stochastic_blockmodel_graph(
                    block_sizes, edge_probs, directed=True
                )
                edge_index = torch.cat([edge_index, curr_edge_index], dim=1)
                t = torch.cat([t, torch.full((curr_edge_index.size(1),), curr_t)])
                edge_probs[i, j + r_1] += p_inc
                curr_t += 1

    return Data(edge_index=edge_index, time=t)


def create_burst(
    n: int = 500, p: float = 0.03, r: int = 10, zipf_a: float = 2.75
) -> Data:
    """
    Create a temporal graph with a bursty pattern.

    Args:
        n (int): Number of nodes.
        p (float): Probability of an edge between two nodes based on G(n,p).
        r (int): Number of repetitions.
        zipf_a (float): Zipf distribution parameter as used in 
            numpy.random.zipf.
    """
    edge_index = erdos_renyi_graph(n, p)
    starting_timestamps = torch.randint(edge_index.shape[1] * r, (edge_index.shape[1],))
    repeated_edge_index = edge_index
    timestamps = starting_timestamps
    for i in range(r):
        pause = random.zipf(zipf_a, edge_index.shape[1])
        curr_timestamps = timestamps[-edge_index.shape[1] :] + torch.tensor(pause)
        timestamps = torch.cat([timestamps, curr_timestamps])
        repeated_edge_index = torch.cat([repeated_edge_index, edge_index], dim=1)

    sorted_idx = torch.argsort(timestamps)
    t = timestamps[sorted_idx]
    edge_index = repeated_edge_index[:, sorted_idx]

    return Data(edge_index=edge_index, time=t)
