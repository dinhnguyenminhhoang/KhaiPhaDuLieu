import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

def load_data():
    print("\n" + "="*60)
    print("BƯỚC 4.1: TRAIN RANDOM FOREST")
    print("="*60)

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} samples")

    with open(PROCESSED_DATA_DIR / 'feature_list.json', 'r') as f:
        feature_list = json.load(f)['features']
    print(f"Features: {len(feature_list)}")

    X_train = train_df[feature_list]
    y_train = train_df['error_type_encoded']
    X_val = val_df[feature_list]
    y_val = val_df['error_type_encoded']

    with open(PROCESSED_DATA_DIR / 'label_encoders.json', 'r') as f:
        encoders = json.load(f)

    label_names = {v: k for k, v in encoders['error_type'].items()}

    return X_train, y_train, X_val, y_val, feature_list, label_names

def evaluate_model(model, X, y_true, label_names, model_name="Model"):
    """Đánh giá model"""
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=[label_names[i] for i in sorted(label_names.keys())],
            zero_division=0,
            output_dict=True
        )
    }

    return results

def train_random_forest_regularized(X_train, y_train, X_val, y_val, label_names):
    print("\nTraining Random Forest...")

    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'max_features': ['sqrt', 0.5],
        'min_impurity_decrease': [0.001, 0.005, 0.01],
        'bootstrap': [True],
        'class_weight': ['balanced']
    }

    base_rf = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=0
    )

    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring='f1_weighted',
        n_jobs=2,
        random_state=RANDOM_SEED,
        verbose=1
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Time: {train_time/60:.2f} min")
    print(f"Best params: {random_search.best_params_}")

    best_model = random_search.best_estimator_

    train_results = evaluate_model(best_model, X_train, y_train, label_names)
    val_results = evaluate_model(best_model, X_val, y_val, label_names)

    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")
    print(f"GAP:   {gap:.4f} ({gap*100:.2f}%)")

    model_file = MODEL_DIR / 'random_forest_regularized.pkl'
    joblib.dump(best_model, model_file)
    print(f"Saved: {model_file}")

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return best_model, val_results, train_time, feature_importance, gap

def main():
    X_train, y_train, X_val, y_val, feature_list, label_names = load_data()

    rf_model, rf_results, rf_time, rf_importance, rf_gap = train_random_forest_regularized(
        X_train, y_train, X_val, y_val, label_names
    )

    results_summary = {
        'model_name': 'Random Forest',
        'validation_accuracy': rf_results['accuracy'],
        'validation_f1_score': rf_results['f1_score'],
        'validation_precision': rf_results['precision'],
        'validation_recall': rf_results['recall'],
        'training_time_seconds': rf_time,
        'overfitting_gap': rf_gap
    }

    results_file = RESULTS_DIR / 'random_forest_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results: {results_file}")

    importance_file = RESULTS_DIR / 'random_forest_feature_importance.csv'
    rf_importance.to_csv(importance_file, index=False)
    print(f"Saved feature importance: {importance_file}")

    print(f"\nCompleted - Val Acc: {rf_results['accuracy']:.4f} | F1: {rf_results['f1_score']:.4f} | Gap: {rf_gap:.4f}")

if __name__ == "__main__":
    main()
