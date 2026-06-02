# DPO Preference Optimization

This repository includes source scripts and benchmark outputs for DPO-based preference optimization experiments.

## Verified Performance Improvements (from training logs)
- `rewards/accuracies`: `0.456 -> 0.635` (`+17.9pp`)
- `rewards/margins`: `0.0015 -> 0.0737` (`+0.0722`)

## Contest Benchmark Snapshot
- KMMLU (non-reasoning): `38.90%`
- MMLU-Redux (non-reasoning): `40.38%`

## Included Files
- DPO training source in `scripts/`
- Benchmark CSV outputs in `results/`
- Core logs in `logs/`
