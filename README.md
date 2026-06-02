# DPO Preference Optimization

This repository includes source scripts, logs, and benchmark outputs for DPO-based preference optimization experiments.

## 1) Dataset and Preference Construction
- Input preference dataset path (training script arg): `--data <preference_data_path>`
- Example run used in logs:
  - total preference pairs: `1058` (`Train: 1005`, `Eval: 53`) in `dpo_train.log`
- Distill+DPO stage log shows larger generated preference set:
  - generated pairs: `2042`
  - used for DPO: `Train: 1939`, `Eval: 103`

## 2) Training Method
- Framework: `trl` `DPOTrainer` + `transformers` + `peft`
- Objective: Direct Preference Optimization with KL control (`beta`)
- Default training setup from `scripts/dpo_train.py`:
  - `lr=5e-7`, `epochs=1` (log run used `epochs=3` in one experiment)
  - `batch_size=2`, `grad_accum=8` (effective batch `16`)
  - `beta=0.1`, `max_length=1024`
  - optimizer schedule: cosine + warmup ratio `0.1`
  - precision: `bf16`
- LoRA option for memory efficiency:
  - `r=32`, `alpha=64`, dropout `0.05`
  - target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## 3) Evaluation Protocol
- Internal training signals: `rewards/accuracies`, `rewards/margins`, loss trends
- Post-train benchmark logs include KMMLU / MMLU-Redux / KMMLU-Redux-LG snapshots
- Saved benchmark CSVs:
  - `results/results_dpo_n50.csv`
  - `results/results_dpo_both_full_n50.csv`
  - `results/results_pruned29L_distill_dpo_n50.csv`

## 4) Verified Performance Improvements (from logs)
| Signal | Early | Late | Gain |
|---|---:|---:|---:|
| rewards/accuracies | 0.456 | 0.635 | +17.9pp |
| rewards/margins | 0.0015 | 0.0737 | +0.0722 |

Contest benchmark snapshot (non-reasoning):
- KMMLU: `38.90%`
- MMLU-Redux: `40.38%`

## 5) Repro Commands
```bash
# basic DPO
python scripts/dpo_train.py --model ./base_model --data ./preference_data

# LoRA DPO
python scripts/dpo_train.py --model ./base_model --data ./preference_data --lora

# custom hyperparams
python scripts/dpo_train.py --model ./base_model --data ./preference_data --lr 5e-7 --epochs 2 --beta 0.1
```

## 6) Repository Contents
- DPO training source in `scripts/`
- Benchmark CSV outputs in `results/`
- Core logs in `logs/`
