# Why volumetric attack detectors fail on unseen networks — code and results

Reproduces every table and figure of the manuscript
"Why volumetric attack detectors fail on unseen networks: Exporter harmonisation,
signature-inversion diagnostics and harm-aware invariant boosting" (A. Hajjouz, E. Avksentieva).

## Files
| File | Purpose |
|---|---|
| `cfs.py` | Loaders for the four networks, Canonical Flow Semantics (CFS), descriptors, BRE, per-network z-score |
| `ergb.py` | Environment-aware histogram gradient boosting: pooled gain, Era-mean, Era-min, Era-dir, **SL-WEG** |
| `run_lodo.py` | Main leave-one-network-out study (10 learners × 4 views × 4 targets × 5 seeds) + in-network reference |
| `run_ablation.py` | BRE contamination, reference size, SL-WEG ρ sweep (`python run_ablation.py contam|nref|rho`) |
| `run_fewshot.py` | Few-shot adaptation study |
| `diagnostics.py` | Univariate AUROC, Direction Consistency Index, source–target signature correlation, Fig. 2 |
| `run_nested_rho.py` | SL-WEG with rho chosen by leave-one-source-out validation (no target information) |
| `synthetic.py` | Planted-shortcut experiment (Table 6) and split activity on real data |
| `analyze.py` | Builds all tables (`results/*.csv`), paired tests, Figs. 1, 3, 4, `results/summary.txt` |
| `data/SHA256SUMS` | Checksums of the exact input files used |

## Data (place in `data/`)
| File name expected | Content | Source used in this study |
|---|---|---|
| `unsw_train.csv`, `unsw_test.csv` | UNSW-NB15 official training/testing sets | github.com/comsyssec/datasets (`unsw-nb15/`) |
| `botiot_full5pc.csv` | Bot-IoT official 5% release, DoS/DDoS + Normal rows (columns proto, dur, spkts, dpkts, sbytes, dbytes, category, subcategory, attack) | github.com/cfoh/IoT-Network-Intrusion-Detection-with-Ensemble-Learners (`UNSW_2018_IoT_Botnet_Full5pc_1-4.zip`) |
| `botiot.csv` | Bot-IoT file containing all 9,543 Normal flows of the complete release | github.com/RSEC-Laboratory/BoostFL-IoT (`data/Bot-IoT.csv`) |
| `toniot_train_test_network.csv` | ToN-IoT Train_Test_Network split (211,043 records) | github.com/thawatchai2799/SecurityArchitectureSmartCityIoT (`data/TON_IoT_Train_Test_Network.csv`) |
| `cicids2017_fri_ddos.csv` | CIC-IDS2017 Friday-afternoon DDoS flows | github.com/StarterArcher/CICIDS2017 |

Official originals: UNSW Canberra (UNSW-NB15, Bot-IoT, ToN-IoT) and the Canadian Institute for Cybersecurity
(CIC-IDS2017). SHA-256 checksums of the exact files used are in `data/SHA256SUMS`.

## Run
```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib scipy joblib
python run_lodo.py --seeds 5 --out lono_main.csv     # ~20 min on 2 cores
python run_fewshot.py cfs_bre
python run_ablation.py contam && python run_ablation.py nref && python run_ablation.py rho
python run_nested_rho.py                              # SL-WEG with rho selected on sources only
python diagnostics.py && python synthetic.py && python analyze.py
```
All randomness is seeded; results in `results/` are the ones reported in the manuscript.
