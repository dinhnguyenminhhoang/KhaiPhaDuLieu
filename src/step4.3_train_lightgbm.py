import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import joblib

import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

def load_data():
    print("\n" + "="*60)
    print("BƯỚC 4.3: TRAIN LIGHTGBM - OPTIMIZED")
    print("="*60)

    print("\n[4.1] Đang load dữ liệu...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    print(f"  ✓ Train set: {len(train_df):,} samples")
    print(f"  ✓ Val set: {len(val_df):,} samples")

    with open(PROCESSED_DATA_DIR / 'feature_list.json', 'r') as f:
        feature_list = json.load(f)['features']

    print(f"  ✓ Features: {len(feature_list)}")

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

def train_lightgbm_optimized(X_train, y_train, X_val, y_val, label_names):
    """LightGBM - Optimized"""
    print("\n[4.4] Training LightGBM (Optimized)...")

    print("\n  📖 STRATEGY:")
    print("     - Balanced regularization")
    print("     - Early stopping")

    params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_samples': 30,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'random_state': RANDOM_SEED,
        'n_jobs': 4,
        'verbose': -1
    }

    print("\n  ⚙️  PARAMETERS:")
    for key, value in params.items():
        print(f"     {key}: {value}")

    start_time = time.time()
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    train_time = time.time() - start_time
    print(f"\n  ⏱️  Time: {train_time:.2f}s")

    train_results = evaluate_model(model, X_train, y_train, label_names)
    val_results = evaluate_model(model, X_val, y_val, label_names)

    print(f"\n  📊 RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")

    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")

    if gap > 0.15:
        print(f"    ⚠️  WARNING: Still overfitting")
    elif gap > 0.10:
        print(f"    ⚠️  CAUTION: Moderate overfitting")
    else:
        print(f"    ✅ GOOD: Low overfitting")

    model_file = MODEL_DIR / 'lightgbm_optimized.pkl'
    joblib.dump(model, model_file)
    print(f"\n  ✓ Saved: {model_file}")

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  📊 TOP 10 IMPORTANT FEATURES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")

    return model, val_results, train_time, feature_importance, gap

def main():
    X_train, y_train, X_val, y_val, feature_list, label_names = load_data()

    print("\n" + "="*60)
    print("BẮT ĐẦU TRAINING LIGHTGBM")
    print("="*60)

    lgb_model, lgb_results, lgb_time, lgb_importance, lgb_gap = train_lightgbm_optimized(
        X_train, y_train, X_val, y_val, label_names
    )

    # Save results to JSON
    results_summary = {
        'model_name': 'LightGBM',
        'validation_accuracy': lgb_results['accuracy'],
        'validation_f1_score': lgb_results['f1_score'],
        'validation_precision': lgb_results['precision'],
        'validation_recall': lgb_results['recall'],
        'training_time_seconds': lgb_time,
        'overfitting_gap': lgb_gap
    }

    results_file = RESULTS_DIR / 'lightgbm_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Saved results: {results_file}")

    # Save feature importance
    importance_file = RESULTS_DIR / 'lightgbm_feature_importance.csv'
    lgb_importance.to_csv(importance_file, index=False)
    print(f"✓ Saved feature importance: {importance_file}")

    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 4.3 - LIGHTGBM!")
    print(f"{'='*60}")
    print(f"\nValidation Accuracy: {lgb_results['accuracy']:.4f}")
    print(f"Validation F1-Score: {lgb_results['f1_score']:.4f}")
    print(f"Overfitting Gap: {lgb_gap:.4f}")

if __name__ == "__main__":
    main()
