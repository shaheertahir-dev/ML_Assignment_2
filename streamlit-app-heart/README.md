# Heart Disease Predictor — Streamlit App

A small web app for students to try the trained heart disease model in a browser, with no install required. Built from the model trained in [`2 - Heart Disease Prediction`](../2%20-%20Heart%20Disease%20Prediction) (the `Heart_Disease_Prediction.ipynb` notebook).

## Deploying to Streamlit Community Cloud (free)

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"Create app"** (or **"New app"**).
3. Choose **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `shaheertahir-dev/ML_Assignment_2`
   - **Branch:** `main`
   - **Main file path:** `streamlit-app-heart/app.py`
5. Under **Advanced settings**, pick **Python 3.11** (the same version as `runtime.txt`).
6. Click **Deploy**. The first deploy takes a minute or two while it installs `requirements.txt`.
7. You'll get a public URL like `https://<something>.streamlit.app`. Paste it into the **Deployment Link** cell at the top of the notebook.

## Running It Locally First (optional, to test changes)

```bash
cd streamlit-app-heart
python -m venv venv
venv\Scripts\activate   # Windows; use "source venv/bin/activate" on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## How It Works

- `app.py` loads `heart_disease_svc_model.pkl` (the same file the notebook produces: a StandardScaler + SVC pipeline) and recreates the exact same `LabelEncoder`s for `sex` and `cp` used during training, so predictions match the notebook exactly.
- Students enter age and maximum heart rate, pick sex and chest pain type from dropdowns, and click **Predict Heart Disease**.
- If you retrain the model in the notebook and want the app to reflect it, copy the new `heart_disease_svc_model.pkl` into this folder and push — Streamlit Cloud will auto-redeploy.
