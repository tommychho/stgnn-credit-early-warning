# ST-GNN for Corporate Credit Default Early Warning

Code release for **"Heterogeneous Spatiotemporal Graph Networks for Corporate Credit Default Early Warning"**, SSTDM-2026, the 21st International Workshop on Spatial and Spatiotemporal Data Mining, held with IEEE ICDM 2026.

The paper is an evaluation-methodology and negative-results study. It reports that the operational metric usually used for credit early warning is confounded by alert volume, replaces it with a per-week capacity rule, and shows through a component audit that four of the model's five components contribute nothing to any score. **No architecture claim in the paper is supported.** This release is scoped to make those results checkable.

## Requirements

```bash
pip install -r requirements.txt
```

**Data:** a WRDS subscription is required to reproduce the paper's numbers. See "Data acquisition" below. A synthetic simulator is included so the pipeline can be exercised without one.

## The evaluation protocol

Two choices decide what a detection rate means. Both are explicit flags in `evaluate.py` rather than implicit defaults, because the paper's main finding is that these choices matter more than any architectural difference it measures.

**Threshold, `--protocol`.** The default `per_week` alerts the top `ceil(c * n_t)` of the `n_t` firms live in week `t`, so the number of files opened is the same every week. `pooled` takes one `(1-c)` quantile over the whole evaluation window instead, which fixes the share of firm-weeks alerted only *on average*: on this panel it averages the intended 14.4 names a week but with a standard deviation of 12.1, a range of 0 to 59, and a lag-1 autocorrelation of +0.993, so the load moves in year-long regimes rather than noise. That is the same defect as a recall target, one level down, which is why the paper reports it as a sensitivity arm. `recall` calibrates each model to a fixed 50% recall target and leaves the alert rate free altogether; under that rule a uniform random scorer reaches 47.6% detection on this panel, the level it reports for trained models.

**Event anchor, `--anchor`.** The default `default` anchors detection on the firm's default date. The alternative `label_onset` anchors on the first snapshot where the horizon label turns on, roughly default minus 52 weeks, which leaves 44 of the 88 defaulters with no lookback to score and caps achievable detection at 50%.

Running with `--protocol recall --anchor label_onset` reproduces the superseded protocol. `evaluate.py` prints the metric ceiling, the number of defaulters any rule could reach, so the cap is visible rather than inferred.

The implementation is in `eval/detection.py`: `per_week_threshold`, `capacity_threshold`, `build_event_anchors`, `detection_rate`, `reachable_defaulters`. `first_detection_lt` and `detection_rate` accept either a float threshold or a snapshot-to-threshold dict, so switching rules is a parameter change rather than a second implementation.

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

Strictly chronological, with a full 52-week embargo between validation and test. Validation is slid back one label horizon and training pays for it, falling from 835 snapshots to 783 so that validation keeps its full 156; the year that falls out between them is 2019.

| Split | Period | Snapshots |
|-------|--------|-----------|
| Train | to 2016-01 | 783 |
| Validation | 2016-01-08 to 2018-12-28 | 156 |
| **Embargo** | **2019-01-04 to 2019-12-27** | **52 (in neither set)** |
| Test | 2020-01-03 to 2024-12-27 | 261 (88 defaulters) |

The 52-week confirmation window means scoring a defaulter early in 2020 needs weeks from before the test boundary: 53 of the 314 scored weeks precede it. Fifty-two of those are the embargo year, fitted on in neither split, and one is the final validation snapshot.

The embargo is necessary because the label looks 52 weeks forward. A validation window ending where testing begins would carry positive labels for firms defaulting after the boundary: auditing the windows against the default dates gives zero contaminated validation positives under the embargo, against 1,030 of 3,059 (33.7%) without it, spanning 35 firms. The split is implemented in `eval/splits.py`.

## Architecture

**This table describes the design as specified, not as trained.** The rightmost column records what the component was measured to contribute once trained; see the audit below.

| Module | Description | Output | Contributes |
|--------|-------------|--------|-------------|
| Mod 1 | Firm LSTM over a 12-quarter history | eps_i in R^128 | yes |
| Mod 2 | Macro gate (5 FRED series) | c_i in R^128 | no |
| Mod 2b | Sector Attention Fusion [SAF variant] | h_fused in R^128 | no |
| Mod 3 | Network systematic pool (4 channels) | m_loc in R^4 | **yes** |
| Mod 4 | GATv2 (2 layers, 4 heads, 64 hidden) | h_GNN in R^64 | no |
| Mod 5 | GRS: gated residual bypass, gate learned per node | h_i in R^64 | no |
| Mod 6 | Classifier trunk [h_i ; m_loc] in R^68 | PD(1q/2q/4q) | n/a |

The gated residual is element-wise and preserves width, so the classifier trunk is 68 for every graph variant and classifier capacity cannot confound the comparison between them. The LSTM-only baseline differs on both counts: it receives no `m_loc` and its trunk takes the 128-dimensional sequence embedding directly, so that one comparison is not input-matched.

GRS was the architectural proposal of the submitted version: a per-node gate intended to route isolated firms around neighbourhood aggregation. Under the optimiser used, the gate was regularised into a firm-independent constant and did no routing. SAF was specified to condition each firm on its sector; its block is annihilated during training and changes no score. Neither is a claim this release supports.

**What the audit found.** Removing each component from a trained checkpoint and re-scoring shows that four of the five contribute nothing measurable. Two mechanisms account for all four, and both follow from applying L2 decay through an adaptive optimiser to every parameter, biases included. Decay on biases drives each sigmoid gate's pre-activation to zero, so the gate rests at the constant sigma(0) = 0.5: the macro gate sits at 0.4996 to 0.5047 across all nine runs, the gated residual at sigma(-0.0016) = 0.4996, and the sector gate's bias decays from -4.0 to exactly 0.0000 in two of three seeds. Decay on weights underflows the two convolutional blocks: the sector embedding reaches about 5e-42, and the denormal share of the GATv2 convolution weights runs from 0.0% to 99.8% across the fifteen checkpoints.

The one component whose removal is detectable is the Module 3 network pool, at -0.0073 +- 0.0047 average precision over nine runs (p = 0.0017). Note that **parameter norms do not diagnose contribution**: a checkpoint at 0.0% denormal still moves AP by +0.0404 when its convolutions are zeroed. Only removal and re-scoring settles it. `training/optim.py` provides the decoupled-decay repair; on a revived branch, deleting the convolutions changes AP by +0.0090 with a standard deviation of 0.0164 across seeds, an interval spanning zero.

## Reproduce the paper's results

### Step 1: train the reported models

```bash
# Sequence baseline, no graph (3 seeds)
python train.py --model lstm  --seed 42  --freq W
python train.py --model lstm  --seed 123 --freq W
python train.py --model lstm  --seed 456 --freq W

# ST-GNN base / + SAF / + GRS, and the no-graph control (3 seeds each)
# `nograph` is trained with every edge tensor empty, so it never sees the graph in training
# OR at inference. It is not a spare ablation: it scores highest of the four, and it is the
# control that tests whether the graph shaped the gates during training, as distinct from
# propagating at inference.
for M in stgnn saf grs nograph; do
  for S in 42 123 456; do python train.py --model $M --seed $S --freq W; done
done

# Ten-seed re-test of the graph gap (paper, "Are Any of the Differences Supported?")
for M in lstm grs saf; do
  for S in 7 99 202 314 555 777 1024; do python train.py --model $M --seed $S --freq W; done
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

- `outputs/results/detection_at_capacity.csv` — AP, detection and lead time at a 1.5% weekly budget under the per-week rule
- `outputs/results/capacity_curve.csv` — the same across budgets from 0.25% to 10%
- `outputs/results/regime_lift.csv` — ranking quality by period, as lift over each period's base rate
- `outputs/results/stats.txt` — seed-level Welch tests under Holm correction
- `outputs/figures/lt_survival.pdf` — lead-time survival curve

To reproduce the paper's Finding 1, that a uniform random scorer matches what the recall-target
rule reports for trained models:

```bash
python evaluate.py --checkpoints outputs/models --protocol recall --random-baseline
```

### Step 3: audit the components

```bash
python component_audit.py --checkpoints outputs/models --data data/processed
```

Removes each component from a trained checkpoint and re-scores, which is what settles whether it
contributes. Produces `outputs/results/component_audit.csv` and `path_ablation.csv`, the two
tables behind the paper's audit. Each removal is chosen to be identity for that block's algebra:
the convolutions, the network pool and the sector block are zeroed, while the macro gate is
frozen at its own time-mean, because zeroing a sigmoid gate gives a constant 0.5 and rescales
every firm rather than removing anything. The script explains this in its header.

To reproduce the superseded protocol instead:

```bash
python evaluate.py --checkpoints outputs/models --protocol recall --anchor label_onset
```

### Step 4: inspect training dynamics (optional)

```bash
python audit_checkpoint.py outputs/models/<checkpoint>.pt
```

Reports parameter statistics for a saved model: the denormal share of the convolution weights, their scale against initialisation, the LayerNorm scales downstream, and the gate's resting value. It needs the checkpoint and nothing else, no data and no GPU.

**Read this as a diagnosis of training dynamics, not as a measure of contribution.** A checkpoint at 0.0% denormal still moves average precision by +0.0404 when its convolutions are zeroed, so weight statistics test the optimiser rather than what a component does. Step 3 is what settles contribution.

## Expected results

Detection under the primary rule, per-week top-k at a 1.5% budget, which is a constant 14 to 15 names a week from about 957 live firms. Three-seed means on the embargoed split.

| Model | DR (%) | seed sd | LT median (w) |
|-------|--------|---------|---------------|
| LSTM-only | 73.1 | 0.7 | 30.7 |
| ST-GNN base | 69.7 | 6.7 | 32.8 |
| ST-GNN + SAF | 76.5 | 5.4 | 31.5 |
| ST-GNN + GRS | 75.4 | 2.4 | 31.7 |

Under `--protocol pooled` the same four report 75.8, 76.1, 79.9 and 79.6 at medians of 33.5 to 35.3 weeks, and the tree baselines appear beside them at 79.9 (XGBoost) and 80.7 (LightGBM). The trees' per-firm detection times were not retained, so they cannot be re-derived under the primary rule and are comparable only with the pooled arm.

**Read this table with its caveat.** The spread across the four variants is no wider than their own seed standard deviations, so the ordering is unstable across seeds and no architecture claim rests on it. ST-GNN base falls below the no-graph sequence model under the deployable rule, though that pair is not input-matched: LSTM-only carries a wider trunk and does not receive the Module 3 pool. Gradient boosting on the same features matches the sequential models on the pooled arm and warns three to four weeks earlier.

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
