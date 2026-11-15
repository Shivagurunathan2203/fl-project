# Dataset Directory for SecureDrive-FL

Place the distracted driver datasets in this directory:

1. **State Farm Distracted Driver Detection**:
   - Path: `data/statefarm/`
   - Should contain:
     - `driver_imgs_list.csv`
     - `imgs/train/c0/` ... `imgs/train/c9/`

2. **AUC Distracted Driver**:
   - Path: `data/auc/`
   - Should contain subject directories or metadata.

If this directory is empty or real datasets are not populated, SecureDrive-FL automatically runs on the Non-IID Dirichlet toy federation without interruption.
