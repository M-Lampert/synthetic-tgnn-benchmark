# Benchmarking Temporal GNNs for Dynamic Link Prediction on Synthetic Temporal Graphs
This repository accompanies the LoG 2024 extended abstract submission "Benchmarking Temporal GNNs for Dynamic Link Prediction on Synthetic Temporal Graphs" and can be used to reproduce the results. It is based on DyGLib - a framework for dynamic link prediction proposed in [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047). The original DyGLib repository can be found [here](https://github.com/yule-BUAA/DyGLib).

## Overview

Dynamic Graph Library (DyGLib) is an open-source toolkit with standard training pipelines, extensible coding interfaces, and comprehensive evaluating strategies, 
which aims to promote standard, scalable, and reproducible dynamic graph learning research. Diverse benchmark datasets and thorough baselines are involved in DyGLib.
![](figures/DyGLib_procedure.jpg)


## Benchmark Datasets and Preprocessing

The four synthetic benchmark datasets that are used in the paper (periodic, burst, triadic, and bipartite) can be found in `DG_data/`. The preprocessing needs to be done as implemented in DyGLib:

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

## Evaluation Tasks

DyGLib supports dynamic link prediction under both transductive and inductive settings with three (i.e., random, historical, and inductive) negative sampling strategies,
as well as dynamic node classification.


## Acknowledgments

We are grateful to the authors of DyGLib for providing an open-source framework for dynamic link prediction
