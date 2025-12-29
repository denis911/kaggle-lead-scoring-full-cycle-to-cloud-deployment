import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import logging

# --- Configuration ---
DATA_PATH = 'Lead Scoring.csv'
MODEL_FILENAME = 'lead_scoring_model.joblib'
RANDOM_STATE = 42
TARGET = 'Converted'
OPTIMAL_THRESHOLD = 0.4817

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_data(path):
    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Replace 'Select' with NaN as found during EDA
    df = df.replace('Select', np.nan)
    return df

def clean_data(df):
    logger.info("Cleaning data and dropping leaky/useless columns...")
    
    # 1. Drop Leaky column identified in EDA
    if 'Tags' in df.columns:
        df.drop('Tags', axis=1, inplace=True)
    
    # 2. Drop unique identifiers
    df.drop(['Prospect ID', 'Lead Number'], axis=1, errors='ignore', inplace=True)
    
    # 3. Drop columns with > 40% missing values (per EDA)
    missing_pct = df.isnull().sum() * 100 / len(df)
    cols_to_drop = missing_pct[missing_pct > 40].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping high-missing columns: {cols_to_drop}")
        df.drop(cols_to_drop, axis=1, inplace=True)
        
    # 4. Feature Pruning: Drop low-variance categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    useless_cols = []
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True, dropna=True)
        if len(counts) <= 1 or counts.iloc[0] > 0.99:
            useless_cols.append(col)
    if useless_cols:
        logger.info(f"Dropping low-variance columns: {useless_cols}")
        df.drop(useless_cols, axis=1, inplace=True)

    # 5. Outlier Capping (99th percentile fallback)
    for col in ['TotalVisits', 'Page Views Per Visit']:
        if col in df.columns:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)

    return df

def consolidate_categories(df):
    logger.info("Consolidating categorical levels...")
    
    # Lead Source consolidation
    if 'Lead Source' in df.columns:
        df['Lead Source'] = df['Lead Source'].replace(['google'], 'Google')
        source_counts = df['Lead Source'].value_counts()
        rare_sources = source_counts[source_counts < 100].index
        df['Lead Source'] = df['Lead Source'].replace(rare_sources, 'Other_Source')

    # Last Activity consolidation
    if 'Last Activity' in df.columns:
        activity_counts = df['Last Activity'].value_counts()
        rare_activities = activity_counts[activity_counts < 100].index
        df['Last Activity'] = df['Last Activity'].replace(rare_activities, 'Other_Activity')

    # Specialization consolidation
    if 'Specialization' in df.columns:
        spec_counts = df['Specialization'].value_counts()
        rare_specs = spec_counts[spec_counts < 100].index
        df['Specialization'] = df['Specialization'].replace(rare_specs, 'Other_Specialization')

    return df

def build_pipeline(X_train):
    logger.info("Building preprocessing pipeline...")
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return full_pipeline

def main():
    # 1. Load & Prep
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = consolidate_categories(df)

    # Impute missing before split (consistent with EDA strategy)
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(TARGET, errors='ignore')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 2. Split
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

    # 3. Train
    pipeline = build_pipeline(X_train)
    logger.info("Training XGBoost model...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluate on Test Set
    logger.info(f"Evaluating model on test set with threshold {OPTIMAL_THRESHOLD}...")
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= OPTIMAL_THRESHOLD).astype(int)

    logger.info("\n" + classification_report(y_test, test_preds))
    logger.info(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    logger.info(f"Test PR-AUC: {average_precision_score(y_test, test_probs):.4f}")

    queue_rate = (test_probs >= OPTIMAL_THRESHOLD).mean()
    logger.info(f"Test Queue Rate: {queue_rate*100:.2f}%")

    # 5. Export
    logger.info(f"Exporting model pipeline to {MODEL_FILENAME}...")
    joblib.dump(pipeline, MODEL_FILENAME)
    logger.info("Retraining script completed successfully.")

if __name__ == "__main__":
    main()
