import pytest
import pandas as pd
import torch

from torch_geometric import seed_everything
from torch_geometric.data import Data

from utils.synthetic import (
    save_graph_to_DG_data, 
    create_periodic,
    create_burst,
    create_triadic,
    create_bipartite
)


def test_save_to_DG_data(tmpdir):
    with tmpdir.as_cwd():
        g = Data(edge_index=torch.tensor([[0, 1], [1, 2]]), time=torch.tensor([0, 1]))
        save_graph_to_DG_data(g, "test")

        df = pd.read_csv("DG_data/test/test.csv", header=0)
        assert df["timestamp"].tolist() == [0, 1]
        assert df["source"].tolist() == [0, 1]
        assert df["destination"].tolist() == [1, 2]
        assert df["state_label"].tolist() == [0, 0]
        assert df["w"].tolist() == [0.0, 0.0]


def test_create_periodic():
    seed_everything(42)
    g = create_periodic(n=3, p=1.0, r=2, poisson_lambda=1)
    assert g.edge_index.shape == (2, 12)
    assert g.time.shape == (12,)
    # Check if the timestamps are increasing
    assert torch.max(g.time) > 6
    # Check if some timestamps repeat
    assert torch.unique(g.time).shape[0] < 12


def test_create_burst():
    seed_everything(42)
    g = create_burst(n=4, p=1.0, r=3, zipf_a=2.75)

    assert g.edge_index.shape == (2, 48)
    assert g.time.shape == (48,)
    # Check if the same edges repeat directly after each other
    assert torch.any((g.edge_index[0, :-1] == g.edge_index[0, 1:]) & (g.edge_index[1, :-1] == g.edge_index[1, 1:]))


def test_create_triadic():
    seed_everything(42)
    g = create_triadic(n=3, p_init=0.5, p_noise=0.0, r=4)

    assert g.edge_index.shape == (2, 5)
    assert g.edge_index.tolist() == [[1, 2, 0, 2, 0], [2, 1, 2, 0, 1]]
    assert g.time.tolist() == [0, 0, 1, 1, 2]


def test_create_bipartite():
    seed_everything(42)
    r_1 = r_2 = 2
    n_cluster = 2
    g = create_bipartite(growth_steps=1, r_1=r_1, r_2=r_2, n_cluster=n_cluster, p_init=0.5, p_inc=0.5)

    expected_edges = 1.5 * (r_1 * r_2) * n_cluster**2
    assert 0.8 * expected_edges <= g.edge_index.shape[1] <= 1.2 * expected_edges
    