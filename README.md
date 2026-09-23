# ST-GNN for Corporate Credit Default Early Warning

Code release for **"Heterogeneous Spatiotemporal Graph Networks for Corporate Credit Default Early Warning"**, SSTDM-2026, the 21st International Workshop on Spatial and Spatiotemporal Data Mining, held with IEEE ICDM 2026.

The paper is an evaluation-methodology and negative-results study. It reports that the operational metric usually used for credit early warning is confounded by alert volume, replaces it with capacity-indexed thresholding, and shows through a checkpoint audit that the message-passing branch of the proposed architecture never trained. **No architecture claim in the paper is supported.** This release is scoped to make those results checkable.

## Requirements

```bash
pip install -r requirements.txt
```

**Data:** a WRDS subscription is required to reproduce the paper's numbers. See "Data acquisition" below. A synthetic simulator is included so the pipeline can be exercised without one.

## The evaluation protocol

Two choices decide what a detection rate means. Both are explicit flags in `evaluate.py` rather than implicit defaults, because the paper's main finding is that these choices matter more than any architectural difference it measures.

**Threshold, `--protocol`.** The default `capacity` sets the operating threshold at the `(1-c)` quantile of the pooled score distribution over the evaluation window, so alerts consume a fixed share of review capacity and the budget is identical across models. The alternative `recall` calibrates each model to a fixed 50% recall target, which leaves the alert rate free. Under that rule a uniform random scorer reaches 47.6% detection on this panel, which is the level it reports for trained models.

**Event anchor, `--anchor`.** The default `default` anchors detection on the firm's default date. The alternative `label_onset` anchors on the first snapshot where the horizon label turns on, roughly default minus 52 weeks, which leaves 44 of the 88 defaulters with no lookback to score and caps achievable detection at 50%.

Running with `--protocol recall --anchor label_onset` reproduces the superseded protocol. `evaluate.py` prints the metric ceiling, the number of defaulters any rule could reach, so the cap is visible rather than inferred.

The implementation is in `eval/detection.py`: `capacity_threshold`, `build_event_anchors`, `detection_rate`, `reachable_defaulters`.

## Data acquisition (requires WRDS access)

The panel covers S&P-rated US non-financial firms, 2001 to 2024, keyed by Compustat `gvkey`. All sources are reachable through a standard WRDS subscription except FRED, which is free.

| Source | Provides | Used for |
|--------|----------|----------|
| Compustat Fundamentals (Quarterly) | Financial-statement items by `gvkey` | 11 firm ratios; deletion codes 02/03 for default labels |
| Compustat / Capital IQ S&P Ratings | Issuer credit-rating history | Numeric rating feature; migration-to-D/SD default trigger |
| CRSP Monthly + CCM Link | Returns, market cap; `permno` to `gvkey` link | Log market cap, 12m return, volatility |
| Mergent FISD | Bankruptcy filings (linked by CUSIP) | Third default-confirmation source |
| FactSet Revere (via WRDS) | Supply-chain and competitor/partner links | Supply-chain + competitor edge channels |
| EX-21 subsidiary filings (SEC) | Parent/subsidiary structure | Corporate-structure edge channel |
| Institutional holdings (13F) | Common blockholders | Common-ownership edge channel |
| FRED (free, no subscription) | 5 macro series (VIX, UNRATE, etc.) | Systematic macro factor |

**All inputs are point-in-time, edges included.** Node features apply rating date, fundamentals quarter-end, CRSP month and macro release on or before the snapshot date, with last-observation-carry-forward for quarterly items. Edges are filtered on their own validity windows: supply-chain on start date, corporate structure on SEC EX-21.1 report date, common ownership on 13F quarter, and competitor and extended supply-chain relations on an active window. No relationship enters a prediction before it was observable.

**Default events** use a composite definition: Compustat deletion codes 02 and 03, S&P rating migration to D or SD, and Mergent FISD bankruptcy filings. Sources are linked by `gvkey`, by the CRSP/Compustat crosswalk, and by CUSIP with a ten-day settlement tolerance. They agree on 81 of the 88 test-window events; the remaining seven come from FISD alone and are retained after confirmation against public filings.

Processed parquet files go under `data/processed/raw/` with this exact layout, read directly by `data/graph_builder.py`:

```
data/processed/raw/
  company/rated_universe.parquet                      # gvkey universe, sector, first/last date
  defaults/compustat_sp_defaults_fallback.parquet     # gvkey, default_date
  ratings/sp_ratings_history.parquet                  # gvkey, rating_date, rating
  financials/ratios_all.parquet                       # gvkey, date, 11 ratios
  stock/crsp_monthly.parquet                          # permno, date, ret, mktcap
  stock/ccm_link.parquet                              # permno <-> gvkey link
  edges/revere_supply_chain_final.parquet             # supplier_gvkey, customer_gvkey, edge_weight, rdate
  edges/parent_subsidiary_edges.parquet               # parent_gvkey, sub_gvkey, rdate
  edges/common_blockholder_edges.parquet              # gvkey_a, gvkey_b, rdate
  edges/revere_extended_relationships.parquet         # gvkey_a, gvkey_b, rel_type, rdate
  macro/macro_factors.parquet                         # quarter-end macro series (or run macro_loader.py)
```

The macro file can be regenerated without WRDS:

```bash
python data/macro_loader.py    # pulls 5 FRED series -> macro/macro_factors.parquet
```

Then build the graph snapshots:

```bash
python data/graph_builder.py ./data/processed/raw/
```

## Data splits

Strictly chronological, with a full 52-week embargo between validation and test paid out of the training window.

| Split | Period | Snapshots |
|-------|--------|-----------|
| Train | to 2016-01 | 783 |
| Validation | 2016-01-08 to 2018-12-28 | 156 |
| Test | 2020 to 2024 | 261 (88 defaulters) |

The embargo is necessary because the label looks 52 weeks forward. A validation window ending where testing begins would carry positive labels for firms defaulting after the boundary: auditing the windows against the default dates gives zero contaminated validation positives under the embargo, against 1,030 of 3,059 (33.7%) without it, spanning 35 firms. The split is implemented in `eval/splits.py`.

## Architecture

| Module | Description | Output |
|--------|-------------|--------|
| Mod 1 | Firm LSTM over a 12-quarter history | eps_i in R^128 |
| Mod 2 | Macro gate (5 FRED series) | c_i in R^128 |
| Mod 2b | Sector Attention Fusion [SAF variant] | h_fused in R^128 |
| Mod 3 | Network systematic pool (4 channels) | m_loc in R^4 |
| Mod 4 | GATv2 (2 layers, 4 heads, 64 hidden) | h_GNN in R^64 |
| Mod 5 [proposed] | GRS: gated residual bypass, learnable gate per node | h_i in R^64 |
| Mod 6 | Classifier trunk [h_i ; m_loc] in R^68 | PD(1q/2q/4q) |

The gated residual is element-wise and preserves width, so the classifier trunk is 68 for every reported variant and classifier capacity cannot confound the comparison between them.

**What the audit found.** Under coupled L2 decay through Adam, the GATv2 branch does not train: 98.9% of its convolution weights reach denormal magnitudes and the LayerNorm scale downstream trains to exactly zero, in all fifteen reported checkpoints. Zeroing the convolutions in a trained checkpoint changes average precision by +0.0000. `training/optim.py` provides the decoupled-decay repair; on a revived branch, deleting the convolutions changes AP by +0.0090 with a standard deviation of 0.0164 across seeds, an interval spanning zero.

## Reproduce the paper's results

### Step 1: train the reported models

```bash
# Sequence baseline, no graph (3 seeds)
python train.py --model lstm  --seed 42  --freq W
python train.py --model lstm  --seed 123 --freq W
python train.py --model lstm  --seed 456 --freq W

# ST-GNN base / + SAF / + GRS (3 seeds each)
for M in stgnn saf grs; do
  for S in 42 123 456; do python train.py --model $M --seed $S --freq W; done
done

# Gradient-boosted tree baselines on the same 11 features (3 seeds each)
for M in xgb lgbm; do
  for S in 42 123 456; do python train.py --model $M --seed $S --freq W; done
done
```

`train.py` also supports `rgcn`, `gatv2`, `ndr` and `coxph`. These are **not reported** in the paper, because they were not re-derived under the corrected protocol. They are left in the code rather than removed.

### Step 2: evaluate

```bash
python evaluate.py --checkpoints outputs/models --output outputs
```

Produces:

- `outputs/results/detection_at_capacity.csv` — AP, detection and lead time at a 1.5% weekly budget
- `outputs/results/capacity_curve.csv` — the same across budgets from 0.25% to 10%
- `outputs/results/regime_lift.csv` — ranking quality by period, as lift over each period's base rate
- `outputs/results/stats.txt` — seed-level Welch tests under Holm correction
- `outputs/figures/lt_survival.pdf` — lead-time survival curve

To reproduce the superseded protocol instead:

```bash
python evaluate.py --checkpoints outputs/models --protocol recall --anchor label_onset
```

### Step 3: audit a checkpoint

```bash
python audit_checkpoint.py outputs/models/<checkpoint>.pt
```

Reports the branch-health statistics behind the convergence audit: the denormal share of the convolution weights, their scale against initialisation, the LayerNorm scales downstream, and the gate's resting value. It needs the checkpoint and nothing else, no data and no GPU, because every quantity is read from saved parameters.

The path ablation quoted in the paper, zeroing the convolutions and re-scoring, needs the evaluation data and so runs through `evaluate.py`.

## Expected results

Detection at a 1.5% weekly review capacity, roughly 14 names per week from about 957 live firms, three-seed means on the embargoed split.

| Model | DR (%) | LT mean (w) |
|-------|--------|-------------|
| LSTM-only | 75.8 | 31.8 |
| ST-GNN base | 76.1 | 32.3 |
| ST-GNN + SAF | 79.9 | 32.4 |
| ST-GNN + GRS | 79.5 | 33.6 |
| XGBoost | 79.9 | 35.4 |
| LightGBM | 80.7 | 35.5 |

**Read this table with its caveat.** The spread across the four sequential variants is smaller than their own seed standard deviations, so the ordering is unstable across seeds and no architecture claim rests on it. Gradient boosting on the same features matches the sequential models and warns three to four weeks earlier. The two families remain inseparable across the whole capacity curve.

Seeds are 42, 123 and 456 for every model, with 7, 99, 202, 314, 555, 777 and 1024 added for the ten-seed comparison reported in the paper. Training takes about 90 minutes per seed on an NVIDIA Tesla T4 over the 783 training snapshots.

## Pre-trained models

Not redistributed. The training data is WRDS-licensed, so checkpoints are not included here. Retrain with `train.py` as above, then run `audit_checkpoint.py` against your own checkpoint.

## Testing without WRDS

The quantitative results depend on WRDS Compustat, a commercial subscription dataset that cannot be redistributed. A synthetic portfolio simulator is included so the pipeline can be run end to end without it. It reproduces the structure of the real data, temporal snapshots, rating migrations, firm entry and exit, sector stress and regime shifts, without exposing any licensed record. It does not copy, sample or approximate any individual WRDS record.

```python
from data.simulator import generate_temporal_training_data
```

Synthetic data exercises the code path and lets you confirm the architecture behaves as described. It cannot reproduce the paper's quantitative results.

## Citation

```bibtex
@inproceedings{ho2026stgnn,
  title     = {Heterogeneous Spatiotemporal Graph Networks for Corporate
               Credit Default Early Warning},
  author    = {Ho, Tommy C.H. and Lee, Vincent C.S. and Perdana, Arif},
  booktitle = {2026 IEEE International Conference on Data Mining Workshops (ICDMW)},
  note      = {21st International Workshop on Spatial and Spatiotemporal Data Mining (SSTDM-2026)},
  year      = {2026},
}
```

## License

MIT. See `LICENSE`.
