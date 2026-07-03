# Online Retail — Daily Sales Forecast (Streamlit)

Predicts total daily sales from order/customer/quantity metrics using a
Linear Regression model trained in `online_Retail_sales_Forecasting_using_ML.ipynb`.

## Files needed in the repo
```
app.py
model.pkl                   # export this from the notebook's last cell
online_retail_cleaned.csv   # the cleaned dataset
requirements.txt
.gitignore
```

> `model.pkl` and the CSV aren't included here — copy them in from your
> local machine/Colab before pushing. If the CSV is large (>100MB), see
> the Git LFS note below.

## 1. Run locally first (recommended before deploying)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## 2. Upload to GitHub

### Option A — brand-new repo
```bash
cd your-project-folder
git init
git add .
git commit -m "Add retail sales forecasting Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### Option B — existing repo
```bash
cd your-project-folder
git add app.py requirements.txt .gitignore README.md
git commit -m "Add Streamlit forecasting app"
git push
```

If prompted for credentials, GitHub no longer accepts your account
password over HTTPS — use a Personal Access Token instead
(GitHub → Settings → Developer settings → Personal access tokens),
or push via SSH.

### Git LFS (only if online_retail_cleaned.csv is large)
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes online_retail_cleaned.csv
git commit -m "Track CSV with LFS"
git push
```

## 3. Deploy for free — Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set **Main file path** to `app.py`.
3. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app automatically.
4. Every future `git push` to that branch redeploys the app.

## Notes
- `app.py` reads the CSV/model from the same folder via relative paths
  (`DATA_PATH`, `MODEL_PATH` at the top of the file) — no code changes
  needed as long as all three files sit together in the repo root.
- If your dataset changes column names, update the column references in
  `compute_daily_metrics()` in `app.py` to match.
