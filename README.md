# LoanLens — Step 4 MVP
**Bharat Horizon Bank — AI-Assisted Retail Home Loan Assessment & Governance**

## 1. Run Locally
1. Install Python 3.10+.
2. Open a terminal in this folder (`loanlens_step4_mvp`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. The application will launch at `http://localhost:8501`.

---

## 2. LLM Policy Explanation Setup (OpenRouter & Gemini)
The local TF-IDF retriever works completely offline without an API key and returns verbatim policy clauses as a transparent extractive fallback.

To enable AI-synthesized policy explanations, you can configure **OpenRouter** (recommended for access to open models and diverse providers) and/or **Google Gemini**.

### Option A: OpenRouter Setup (Recommended)
OpenRouter provides an OpenAI-compatible API (`https://openrouter.ai/api/v1`) giving access to models from Meta (Llama), Google, DeepSeek, Anthropic, and OpenAI.

1. Obtain an API key from [openrouter.ai/keys](https://openrouter.ai/keys).
2. Configure your key and chosen model using either method below:

**Method 1: `.streamlit/secrets.toml` (Persistent & Local)**
Create or edit `.streamlit/secrets.toml` inside `loanlens_step4_mvp/`:
```toml
LLM_PROVIDER = "openrouter"
OPENROUTER_API_KEY = "sk-or-v1-your-openrouter-key"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
```
*(You can use any model identifier from [openrouter.ai/models](https://openrouter.ai/models), such as `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `deepseek/deepseek-chat`, or `openai/gpt-4o-mini`)*.

**Method 2: Environment Variables (PowerShell)**
```powershell
$env:LLM_PROVIDER="openrouter"
$env:OPENROUTER_API_KEY="sk-or-v1-your-openrouter-key"
$env:OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct:free"
streamlit run app.py
```

---

### Option B: Google Gemini Setup
1. Obtain an API key from [aistudio.google.com](https://aistudio.google.com).
2. Add to `.streamlit/secrets.toml`:
```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODEL = "gemini-flash-lite-latest"
```
*(Or set in PowerShell via `$env:GEMINI_API_KEY="AIzaSy..."`)*.

---

### Switching Providers in the App
On **Screen 3 (Policy Explanation)**:
- Open the **"⚙️ LLM Provider & Model Settings"** panel.
- Switch between **OpenRouter** and **Gemini (Google)** via the dropdown.
- You can change the model identifier directly in the input box without restarting the application or modifying code.

---

## 3. Public Deployment (Streamlit Community Cloud)
1. Initialize Git and push `loanlens_step4_mvp` to a private or public GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy LoanLens MVP"
   git remote add origin https://github.com/YOUR_USER/loanlens.git
   git push -u origin main
   ```
2. Open [share.streamlit.io](https://share.streamlit.io) and deploy the repository with `app.py` as the entry point.
3. In your app dashboard under **App Settings → Secrets**, configure your keys:
   ```toml
   LLM_PROVIDER = "openrouter"
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
   
   # Optional Gemini backup
   GEMINI_API_KEY = "AIzaSy..."
   ```

---

## 4. Key Repository Files
- `app.py`: 4-screen Streamlit application with interactive dashboard and HITL review.
- `policy_rag.py`: Local TF-IDF chunk retriever and dual-provider (OpenRouter / Gemini) generator with sanitised extractive fallback.
- `lending_policy.md`: Bharat Horizon Bank lending policy guidelines (`BHB-HL-001`).
- `logistic_regression_pipeline.joblib`: Serialized Scikit-Learn predictive model pipeline.
- `bharat_horizon_home_loan_dataset_v2.xlsx`: Benchmark synthetic retail home loan dataset.
- `review_log.csv`: Persistent audit log for human underwriter decisions and override justifications.
- `requirements.txt`: Application Python dependencies.

---

## 5. Governance & Educational Notice
- This application is an educational prototype developed for Bharat Horizon Bank.
- Synthetic training data labels reflect historical underwriting patterns, not repayment certainty.
- High-risk or borderline credit decisions require mandatory human review.
