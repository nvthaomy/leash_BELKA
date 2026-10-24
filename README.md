# Leash Bio BELKA — binding prediction

Code, run records and selected models for the Kaggle
[Leash Bio – Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA)
competition: predict whether a small molecule binds BRD4, HSA or sEH from its SMILES.

Approach: Morgan fingerprints (256–2048 bits) fed to fully connected nets (NN, NN2–NN5),
a 1D CNN, random forests and graph conv nets (GCN/FPGCN), trained on class-rebalanced
subsets of the 295M-row training set (5k to 15M rows), mostly one model per protein.
Training ran on the USC CARC (SLURM) cluster; `cluster/` mirrors that working directory.

## Best results (private leaderboard)

| Model | Fingerprint | Train size | Pos. rate | Private | Folder |
|---|---|---|---|---|---|
| **NN4** | 1024 | 5M | 0.05 | **0.220** | `cluster/nn_morgan-1024_5M` |
| NN4 | 1024 | 10M | 0.05 | 0.197 | `cluster/nn_morgan-1024_10M` |
| NN5 | 1024 | 10M | 0.041 | 0.197 | `cluster/split2_nn_morgan` |
| NN5 | 1024 | 10M | 0.05 | 0.196 | `cluster/nn_morgan-1024_10M` |
| NN5 | 1024 | 5M | 0.05 | 0.194 | `cluster/nn_morgan-1024_5M` |
| NN5 | 1024 | 5M | 0.082 | 0.192 | `cluster/split2_nn_morgan` |
| NN4 | 1024 | 15M | 0.027 | 0.190 | `cluster/split2_nn_morgan` |
| NN4 | 1024 | 1M | 0.05 | 0.188 | `cluster/nn_morgan-1024_1M` |
| CNN1d | 1024 | 5M | 0.05 | 0.185 | `cluster/random_split_nn_morgan-1024` |
| FPGCN | 1024 | 5k | 0.05 | 0.015 | `cluster/gcn_morgan-1024` |

NN4 with 2048 bits on 5M rows (`cluster/nn_morgan-2048_5M`) had the best public score (0.542).
The full experiment log, including the earlier public-score table, is in `summary.xlsx`.

## Layout

- `cluster/scripts/` — latest shared code (`nn_v3.py`, `tools.py`, `write_submission.py`).
- `cluster/<run family>/` — one folder per experiment family: the training script used
  (`nn.py`/`nn_v2.py`/`rf.py`), SLURM scripts (`run_*.sh`, `job`), logs (`*.out`) and one
  `checkpoints_*` folder per model with `hyperparameters.txt`, `results*.txt` and plots.
- `cluster/data/` — scripts that build the training subsets and fingerprints
  (`split2/sample_train.py`, `random_splits/sample_train_independent.py`, `generate_fp/`).
- `models/`, `transformation/`, `icm/` — early local notebooks and ICM docking setup.

## What is in git and what isn't

Kept: all code, configs, logs, plots and result files for every run; model weights
(`model.pth` / `model_best_fold-*.pth`) and final submission CSVs only for the runs in the
table above.

Not kept (about 300 GB; listed with sizes in `MANIFEST_excluded.txt`):
- the Kaggle data (`train.parquet`, `test.parquet`, `sample_submission.csv`) — download
  again with `kaggle competitions download -c leash-BELKA`;
- derived training subsets, fingerprint and graph pickles (`*.parquet`, `*.pkl`), ICM
  docking output (`*.sdf`, `*.log`);
- per-fold test predictions, resume checkpoints (`model_checkpoint*.pth`) and the weights
  and submissions of weaker runs.

## Reproducing a run

1. Download the competition data into `data/`.
2. Build a training subset with the sampling scripts in `cluster/data/`.
   **The subsampling calls `DataFrame.sample` without a seed**, so a rerun draws a
   different subset than the one behind the scores above.
3. Train each protein, e.g. for the best run (`cluster/nn_morgan-1024_5M/run_brd4.sh`):
   `python -u nn.py BRD4 -model_name NN4 -checkpoint <1M NN4 weights> -model_index 10-v1`,
   and likewise for HSA and sEH. That run was warm-started from the 1M NN4 model; its
   `checkpoint_epoch_30.pth` is gone, and the closest saved weights are
   `cluster/nn_morgan-1024_1M/checkpoints_NN4_*_10_*/model.pth`.
4. Combine the three per-protein predictions with `write_submission.py`.

To score with a saved model instead of retraining, load `model.pth` with the matching
model class (`Net4`, `Net5`, … in that folder's `tools.py`) and the fingerprint size in
the folder name.
