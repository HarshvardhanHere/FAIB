# LoanLens — Step 4 MVP

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. The browser should open the app.

## Optional LLM explanation
The policy retriever works without an API key and returns retrieved policy sections as an extractive fallback.
To enable Gemini-generated explanations, configure `GEMINI_API_KEY` in your local environment or in Streamlit Community Cloud app settings under Secrets:
```toml
GEMINI_API_KEY = "your-key"
```
Never commit secrets to GitHub or include them in prompt logs. If using a free API tier, check the provider's current quota, availability, and terms.

## Public deployment (Streamlit Community Cloud)
1. Create a GitHub repository and upload the contents of this folder.
2. Do not upload any API keys.
3. Open Streamlit Community Cloud and deploy the repository, selecting `app.py` as the entry point.
4. Add `GEMINI_API_KEY` under app secrets if you want LLM-generated explanations.
5. Test the public URL using synthetic demo cases.

## Files
- `app.py`: four-screen Streamlit app.
- `policy_rag.py`: local retrieval and optional Gemini generation.
- `lending_policy.md`: fictional policy source.
- `logistic_regression_pipeline.joblib`: model trained in Step 2.
- `bharat_horizon_home_loan_dataset_v2.xlsx`: synthetic dataset for display/evaluation context.
- `requirements.txt`: app dependencies.

## Current limitations
- Dataset labels are synthetic and intentionally noisy.
- FOIR uses the illustrative 8.5% annual interest rate from the demo dataset.
- The model's probability is not a repayment probability.
- Policy checks are simplified and must be tested against the team's agreed policy.
- The reviewer log is local CSV and may not persist on hosted services.
- This app is not a real lending system or financial advice.
