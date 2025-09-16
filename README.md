# Economic Crisis Detection (Streamlit)

Detect unemployment/GDP anomalies by country and year using Isolation Forest or LOF.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to a new GitHub repo.
2. In Streamlit Cloud, create a new app by selecting your repo and `app.py`.
3. No secrets required.
4. The app will boot and load `sample_data.csv` unless you upload your own file.

## Data format
Required columns:
- Country Name
- Year
- Employment Sector: Agriculture
- Employment Sector: Industry
- Employment Sector: Services
- Unemployment Rate
- GDP (in USD)
