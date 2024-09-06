# Benchmarking Temporal GNNs for Dynamic Link Prediction on Synthetic Temporal Graphs
This repository accompanies the LoG 2024 extended abstract submission "Benchmarking Temporal GNNs for Dynamic Link Prediction on Synthetic Temporal Graphs" and can be used to reproduce the results. It is based on DyGLib - a framework for dynamic link prediction proposed in [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047). The original DyGLib repository can be found [here](https://github.com/yule-BUAA/DyGLib).

## Running the Experiments

We provide a `bash`-script to run our experiments as follows:
```sh
bash run_experiments.sh
```
Using Windows, we recommend to use the `git`-bash to run the script.

## Setup

We provide a ready-to-use [DevContainer](https://microsoft.github.io/vscode-essentials/en/09-dev-containers.html) that installs everything that is necessary to use this repository or reproduce our results. You can also install the dependencies via `pip` by running:
```sh
pip install torch==2.2.0
pip install -r requirements.txt
```
Note that these instructions assume a Linux system with a NVIDIA GPU installed. Please refer to the installation manual of [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for further details on how to install it on other systems and modify `requirements.txt` accordingly.

### Running DevContainer without GPU

If you want to use the DevContainer but do not have a NVIDIA GPU, you need to comment out the lines 36 and 37 of your `.devcontainer/devcontainer.json`.

## Benchmark Datasets and Preprocessing

The four synthetic benchmark datasets that are used in the paper (periodic, burst, triadic, and bipartite) can be found in `DG_data/`. Note that we also provide methods to generate additional synthetic networks with different parameters (see `datasets.ipynb` for examples). The preprocessing needs to be done as implemented in DyGLib:

Run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *burst* dataset, run the following commands:
```{bash}
python preprocess_data.py --dataset_name burst
```

## Dynamic Graph Learning Models

Nine popular continuous-time dynamic graph learning methods are included in DyGLib, including 
[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895), 
[DyRep](https://openreview.net/forum?id=HyePrhR5KX), 
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH), 
[TGN](https://arxiv.org/abs/2006.10637), 
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj), 
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg), 
[TCL](https://arxiv.org/abs/2105.07944),
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), and [DyGFormer](https://arxiv.org/abs/2303.13047).

## Testing Synthetic Generation Methods

We include unit-tests for the generation methods of the proposed synthetic networks. You can run them as follows:
```
pytest
```

## Acknowledgments

We are grateful to the authors of DyGLib for providing an open-source framework for dynamic link prediction
