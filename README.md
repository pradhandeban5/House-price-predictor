# Hearth — House Price Intelligence

A single-site house price prediction dashboard built with Streamlit. The complete
application and its supporting code live in `public/`; Streamlit handles both
the interactive UI and model execution.

## What it does

- Loads `.xlsx` and `.xlsm` workbooks with pandas and openpyxl.
- Uses `public/data/Housing.xlsx` when present (including the provided
  lower-case `price` target); otherwise creates
  `public/data/house_prices.xlsx` with reproducible sample data on first run.
- Accepts uploaded workbooks with a numeric `Price` target (matched
  case-insensitively).
- Presents a prediction-only house-details form; model configuration stays out
  of the customer-facing page.
- Imputes numeric values with the median and categorical values with the most
  frequent value; scales numeric columns and one-hot encodes categorical ones.
- Evaluates model quality privately during fitting, but does not show training
  details or performance metrics in the customer-facing page.
- Saves the current model to `public/models/pipeline.joblib`. The dataset,
  selected features, model settings, and scikit-learn version are fingerprinted
  so a matching pipeline can be reused. Changing data, features, or settings
  results in a new fit; the sidebar button also forces a retrain.
- Provides clearly labeled numeric controls and dropdowns populated with
  observed property options, followed by a prominent estimated value.

## Run locally

Use Python 3.11 (validated here; Python 3.10–3.12 are supported). From the
project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run public/app.py
```

If PowerShell blocks environment activation, invoke the environment's Python
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run public/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
The first start trains the model on `public/data/Housing.xlsx` if present;
otherwise, it creates the sample workbook and fits the model. In the provided
housing dataset, `price` is the target, continuous fields such as `area` are
numeric controls, small integer fields such as bedrooms/stories/parking use
dropdowns, and yes/no or furnishing fields use dropdowns populated from the
workbook. Customers only see the property form and resulting estimate. The
predicted value uses the same units as the dataset because the workbook does
not specify a currency.

## Deploy as one Streamlit site

The repository root is the deployment root. Do not split the application into
separate backend and frontend services.

### Streamlit Community Cloud

1. Push this project to a GitHub repository with `requirements.txt`,
   `README.md`, and the `public/` directory at its root.
2. Create a Community Cloud app for that repository and set the **Main file
   path** to `public/app.py`.
3. Deploy. The app generates its default workbook and model on first use if
   they are absent.

### Render

1. Create a Python web service from the repository root.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to
   `streamlit run public/app.py --server.address 0.0.0.0 --server.port $PORT`.
4. Use a persistent disk if generated data and model files must survive service
   restarts. Otherwise the app safely regenerates/retrains them when needed.

The same single-entrypoint pattern works on Hugging Face Spaces (Streamlit
SDK) and Railway: install `requirements.txt` and run
`streamlit run public/app.py` on the platform-provided host and port.

## Project layout

```text
.
├── requirements.txt
├── README.md
└── public/
    ├── app.py
    ├── custom_style.css
    ├── data_handler.py
    ├── model_engine.py
    ├── data/       # generated house_prices.xlsx
    └── models/     # generated pipeline.joblib
```
