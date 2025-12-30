# Lead Scoring System: From CRM Mess to Production API

This project demonstrates a full-cycle machine learning pipeline designed to solve a common business problem: **converting sales leads**. 

The goal is to identify "Hot Leads" from a typical CRM data dump (containing missing values, high-cardinality categorical features, and noise) so that a sales team can prioritize their calls and achieve a target precision of ~80%.

## 🚀 Quick Start (Local Development)

This project uses `uv` for lightning-fast, reproducible Python environment management.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```
2.  **Run the analysis**:
    ```bash
    uv run jupyter notebook EDA.ipynb
    ```
3.  **Train the production model**:
    ```bash
    uv run python train.py
    ```

## 🧠 Technical Workflow & Key Decisions

### 1. Data Cleaning & Exploratory Data Analysis (EDA)
CRM data is notorious for being "messy." We took several strategic decisions during EDA:
- **Handling 'Select'**: In this dataset, 'Select' in categorical fields was equivalent to a `Null`. We treated these as missing values.
- **Noise Reduction**: We identified columns with >40% missing data or extreme skew (>99% dominant class) and pruned them to prevent the model from learning noise.
- **Categorical Consolidation**: Features like `Lead Source` and `Last Activity` had dozens of rare categories. We consolidated anything under 1% frequency into "Other" buckets. This reduces sparsity and makes the model more robust.
- **Leakage Prevention**: We dropped the `Tags` column. While highly predictive, tags are often added *after* a lead converts, which is a classic form of "data leakage" that wouldn't be available in real-time.

### 2. Modeling Strategy
We compared three approaches using a **70/15/15** Train/Validation/Test split:
- **Logistic Regression**: Our baseline for explainability.
- **Random Forest**: To capture non-linear interactions.
- **XGBoost**: Our final choice. It handled the remaining missing values and complex relationships most effectively, achieving a **Test ROC-AUC of 0.90**.

### 3. Business-First Threshold Tuning
A standard 0.5 threshold is rarely ideal for business. As requested by the CEO, we prioritized **Precision**.
- **The Threshold**: We used Yellowbrick's `DiscriminationThreshold` to find the optimal cut-off of **0.4817**.
- **The Result**: This yields an expected **80% precision**—meaning 4 out of 5 leads flagged by the system are likely to convert.
- **Efficiency**: This results in a **~38% Queue Rate**, allowing the sales team to ignore the bottom 60% of cold leads entirely.

### 4. Production Pipeline
We refactored the experimental code into a modular `train.py` script.
- **Pipelines**: We used `scikit-learn` Pipelines to bundle the `StandardScaler`, `OneHotEncoder`, and `XGBClassifier` into a single `.joblib` file. This ensures that the exact same transformations are applied during training and inference (no "training-serving skew").

## 🛠️ Serving & Deployment

### Local Inference
The model is served via a **FastAPI** application (`app.py`).
- **Pydantic Validation**: Every request is strictly validated against the expected 13 features.
- **Speed**: Built on Uvicorn for high-concurrency performance.

**Testing Locally**:
```bash
# Start server
uv run uvicorn app:app --reload

# Run test script
uv run python local_test.py
```

### Dockerization
The app is containerized for "run anywhere" reliability.
- **Optimized Build**: We used a multi-stage Docker build to keep the final image slim (~150MB), only including the necessary virtual environment and artifacts.
- **Testing Docker**:
  ```bash
  docker build -t lead-scoring-app .
  docker run -p 8000:8000 lead-scoring-app
  ```

### Cloud Deployment (Render)
The system is live at: `https://kaggle-lead-scoring-full-cycle-to-cloud.onrender.com`

**Verify Cloud Service**:
```bash
uv run python cloud_test.py
```

> [!NOTE]
> **Cold Starts**: Since this is running on a free Render instance, the service spins down after inactivity. The first request might take 30-50 seconds to "wake up" the container. Subsequent requests are near-instant.

## 📈 Business Value
This project moves beyond "just a model" to a "data product." By converting a messy CRM dump into a clean, containerized API, it provides a direct bridge between data science and sales operations. The ability to guarantee a specific precision level (80%) allows management to confidently allocate human resources to the leads that matter most.
