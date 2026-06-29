# AdvectNet: Satellite Temporal Super-Resolution

AdvectNet enhances the temporal resolution of geostationary satellite imagery. The confirmed submission model uses frozen pretrained RAFT for optical flow and trains a lightweight time-conditioned U-Net on Himawari-9 10-minute thermal infrared imagery using reconstruction, gradient, advection-consistency, and sparse-source losses. The trained model is then tested for transfer to INSAT-3DR TIR1 data from MOSDAC.

## Problem

INSAT imagery is often available at a coarser temporal cadence than the cloud motion we want to observe. Given endpoint frames

```text
X0 = image at t
X3 = image at t + 30 min
```

the model generates intermediate frames

```text
Y1_hat = generated t + 10 min
Y2_hat = generated t + 20 min
```

and is validated on Himawari-9 where the true `Y1` and `Y2` frames exist.

## Repository Structure

```text
.
|-- data_sample/                 # tiny sample metadata/placeholders only
|-- notebooks/                   # Kaggle notebooks for training and testing
|-- src/
|   |-- preprocess.py            # Himawari/MOSDAC preprocessing helpers
|   |-- train.py                 # training utilities and checkpoint helpers
|   |-- inference.py             # tiled inference helpers
|   |-- metrics.py               # MSE/PSNR/SSIM helpers
|   `-- streamlit_app.py         # minimal demo app skeleton
|-- models/                      # final checkpoints go here locally, not committed
|-- results/                     # evaluation CSVs/images/GIFs go here locally
|-- README.md
`-- requirements.txt
```

## Main Notebooks

- `notebooks/advectnet_himawari_prototype.ipynb`  
  Himawari prototype with validation against actual `t+10` and `t+20`, including physics-guided advection/source regularization.

- `notebooks/himawari9_month_raft_finetune_kaggle.ipynb`  
  Month-scale training notebook. The confirmed path is Stage 1: frozen RAFT + trainable U-Net with physics-guided losses. Stage 2 partial RAFT fine-tuning is experimental and should not be reported as final unless its validation results are confirmed.

- `notebooks/insat3dr_tir1_mosdac_advectnet_test_kaggle.ipynb`  
  MOSDAC INSAT-3DR TIR1 download and transfer testing notebook.

- `notebooks/insat_mosdac_fog_advectnet_test_kaggle.ipynb`  
  Earlier categorical fog-product transfer experiment.

- `notebooks/mosdac_api_reference.ipynb`  
  Reference notebook for MOSDAC `mdapi.py` access.

## Checkpoints

Large checkpoints are ignored by git. Put trained files locally in `models/`, for example:

```text
models/advectnet_unet_himawari.pth
```

For Kaggle, save the checkpoint as notebook output or upload it as a Kaggle dataset.

## Minimal Kaggle Workflow

1. Run `notebooks/himawari9_month_raft_finetune_kaggle.ipynb`.
2. Save the confirmed frozen-RAFT/U-Net checkpoint:

```text
/kaggle/working/hima_month_checkpoints/advectnet_stage1_epXXX.pth
```

3. Attach the checkpoint and MOSDAC helper dataset to the INSAT notebook.
4. Run `notebooks/insat3dr_tir1_mosdac_advectnet_test_kaggle.ipynb`.

## Notes

- Himawari validation is the correct source for true `t+10` and `t+20` error because Himawari has 10-minute cadence.
- INSAT-3DR TIR1 transfer testing usually has 30-minute cadence; it can generate `+10/+20`, but true ground truth may not exist in the downloaded INSAT sequence.
- MOSDAC credentials must not be committed. Use Kaggle secrets or local environment variables.
- RAFT fine-tuning is not yet part of the confirmed architecture. Report the confirmed architecture as frozen RAFT + trained U-Net unless the fine-tuned checkpoint is fully validated.
- Physics-guided loss terms are part of the confirmed U-Net training objective. The advection term is a motion-consistency regularizer based on frozen RAFT intermediate flow, not a claim of exact atmospheric wind recovery.
