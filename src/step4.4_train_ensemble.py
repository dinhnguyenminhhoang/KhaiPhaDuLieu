import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import joblib

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

def load_data():
    print("\n" + "="*60)
    print("BƯỚC 4.4: TRAIN ENSEMBLE - WEIGHTED VOTING")
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

def load_trained_models():
    """Load các models đã train từ các bước trước"""
    print("\n[4.2] Đang load các models đã train...")

    models = {}

    # Load Random Forest
    rf_file = MODEL_DIR / 'random_forest_regularized.pkl'
    if rf_file.exists():
        models['rf'] = joblib.load(rf_file)
        print(f"  ✓ Loaded Random Forest: {rf_file}")
    else:
        raise FileNotFoundError(f"Random Forest model not found: {rf_file}")

    # Load XGBoost
    xgb_file = MODEL_DIR / 'xgboost_regularized.pkl'
    if xgb_file.exists():
        models['xgb'] = joblib.load(xgb_file)
        print(f"  ✓ Loaded XGBoost: {xgb_file}")
    else:
        raise FileNotFoundError(f"XGBoost model not found: {xgb_file}")

    # Load LightGBM
    lgb_file = MODEL_DIR / 'lightgbm_optimized.pkl'
    if lgb_file.exists():
        models['lgb'] = joblib.load(lgb_file)
        print(f"  ✓ Loaded LightGBM: {lgb_file}")
    else:
        raise FileNotFoundError(f"LightGBM model not found: {lgb_file}")

    # Load results for weights calculation
    results = {}
    for name, file_name in [('rf', 'random_forest_results.json'),
                            ('xgb', 'xgboost_results.json'),
                            ('lgb', 'lightgbm_results.json')]:
        result_file = RESULTS_DIR / file_name
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"  ⚠️  Warning: {file_name} not found, will use equal weights")
            results[name] = {'validation_f1_score': 1.0}

    return models, results

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

def train_ensemble(models, model_results, X_train, y_train, X_val, y_val, label_names):
    """Voting Ensemble - Weighted by F1"""
    print("\n[4.5] Training Ensemble (Weighted Voting)...")

    print("\n  📖 STRATEGY:")
    print("     - Soft voting từ 3 models")
    print("     - Weights dựa trên validation F1-score")

    # Calculate weights based on F1 scores
    weights = []
    for name in ['rf', 'xgb', 'lgb']:
        f1 = model_results[name]['validation_f1_score']
        weights.append(f1)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]

    print(f"\n  ⚙️  WEIGHTS:")
    model_names = ['Random Forest', 'XGBoost', 'LightGBM']
    for name, weight in zip(model_names, weights):
        print(f"     {name}: {weight:.3f}")

    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models['rf']),
            ('xgb', models['xgb']),
            ('lgb', models['lgb'])
        ],
        voting='soft',
        weights=weights,
        n_jobs=4
    )

    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\n  ⏱️  Time: {train_time:.2f}s")

    # Evaluate
    train_results = evaluate_model(voting_clf, X_train, y_train, label_names)
    val_results = evaluate_model(voting_clf, X_val, y_val, label_names)

    print(f"\n  📊 RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")

    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")

    if gap > 0.15:
        print(f"    ⚠️  WARNING: Overfitting")
    elif gap > 0.10:
        print(f"    ⚠️  CAUTION: Moderate overfitting")
    else:
        print(f"    ✅ GOOD: Low overfitting")

    # Compare with individual models
    print(f"\n  📈 COMPARISON:")
    print(f"    Random Forest Val F1: {model_results['rf']['validation_f1_score']:.4f}")
    print(f"    XGBoost Val F1:       {model_results['xgb']['validation_f1_score']:.4f}")
    print(f"    LightGBM Val F1:      {model_results['lgb']['validation_f1_score']:.4f}")
    print(f"    Ensemble Val F1:      {val_results['f1_score']:.4f}")

    improvement = val_results['f1_score'] - max(
        model_results['rf']['validation_f1_score'],
        model_results['xgb']['validation_f1_score'],
        model_results['lgb']['validation_f1_score']
    )

    if improvement > 0:
        print(f"    ✅ Ensemble BETTER by {improvement:.4f}")
    else:
        print(f"    ⚠️  Ensemble WORSE by {abs(improvement):.4f}")

    # Save
    model_file = MODEL_DIR / 'ensemble_weighted.pkl'
    joblib.dump(voting_clf, model_file)
    print(f"\n  ✓ Saved: {model_file}")

    return voting_clf, val_results, train_time, gap

def main():
    X_train, y_train, X_val, y_val, feature_list, label_names = load_data()

    # Load trained models
    models, model_results = load_trained_models()

    print("\n" + "="*60)
    print("BẮT ĐẦU TRAINING ENSEMBLE")
    print("="*60)

    ensemble_model, ensemble_results, ensemble_time, ensemble_gap = train_ensemble(
        models, model_results, X_train, y_train, X_val, y_val, label_names
    )

    # Save results to JSON
    results_summary = {
        'model_name': 'Ensemble',
        'validation_accuracy': ensemble_results['accuracy'],
        'validation_f1_score': ensemble_results['f1_score'],
        'validation_precision': ensemble_results['precision'],
        'validation_recall': ensemble_results['recall'],
        'training_time_seconds': ensemble_time,
        'overfitting_gap': ensemble_gap
    }

    results_file = RESULTS_DIR / 'ensemble_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Saved results: {results_file}")

    # Create comparison table
    print("\n" + "="*60)
    print("SO SÁNH TẤT CẢ MODELS")
    print("="*60)

    comparison = []
    for name, result in [('Random Forest', model_results['rf']),
                        ('XGBoost', model_results['xgb']),
                        ('LightGBM', model_results['lgb']),
                        ('Ensemble', results_summary)]:
        comparison.append({
            'Model': name,
            'Val Accuracy': result['validation_accuracy'],
            'Val F1-Score': result['validation_f1_score'],
            'Overfitting Gap': result['overfitting_gap'],
            'Training Time (s)': result['training_time_seconds']
        })

    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('Val F1-Score', ascending=False)

    print("\n📊 BẢNG SO SÁNH:")
    print(df_comparison.to_string(index=False))

    # Find best model
    df_comparison['score'] = df_comparison['Val F1-Score'] - (df_comparison['Overfitting Gap'] * 0.5)
    best_idx = df_comparison['score'].idxmax()
    best_model = df_comparison.loc[best_idx, 'Model']

    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Val F1: {df_comparison.loc[best_idx, 'Val F1-Score']:.4f}")
    print(f"   Gap: {df_comparison.loc[best_idx, 'Overfitting Gap']:.4f}")

    comparison_file = RESULTS_DIR / 'model_comparison_all.csv'
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n✓ Saved comparison: {comparison_file}")

    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 4.4 - ENSEMBLE!")
    print(f"{'='*60}")
    print(f"\nValidation Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"Validation F1-Score: {ensemble_results['f1_score']:.4f}")
    print(f"Overfitting Gap: {ensemble_gap:.4f}")

if __name__ == "__main__":
    main()
