import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    with open(PROCESSED_DATA_DIR / 'feature_list.json', 'r') as f:
        feature_list = json.load(f)['features']
    
    X_train = train_df[feature_list]
    y_train = train_df['error_type_encoded']
    X_val = val_df[feature_list]
    y_val = val_df['error_type_encoded']
    
    with open(PROCESSED_DATA_DIR / 'label_encoders.json', 'r') as f:
        encoders = json.load(f)
    
    label_names = {v: k for k, v in encoders['error_type'].items()}
    
    return X_train, y_train, X_val, y_val, feature_list, label_names

def evaluate_model(model, X, y_true, label_names, model_name="Model"):
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
    
    best_model = random_search.best_estimator_
    
    train_results = evaluate_model(best_model, X_train, y_train, label_names)
    val_results = evaluate_model(best_model, X_val, y_val, label_names)
    
    gap = train_results['accuracy'] - val_results['accuracy']
    
    model_file = MODEL_DIR / 'random_forest_regularized.pkl'
    joblib.dump(best_model, model_file)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, val_results, train_time, feature_importance, gap

def train_xgboost_regularized(X_train, y_train, X_val, y_val, label_names):
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [5, 10, 15],
        'gamma': [0.1, 0.3, 0.5],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [2, 3, 5]
    }
    
    base_xgb = xgb.XGBClassifier(
        tree_method='hist',
        random_state=RANDOM_SEED,
        n_jobs=4,
        eval_metric='mlogloss'
    )
    
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='f1_weighted',
        n_jobs=2,
        random_state=RANDOM_SEED,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    
    train_results = evaluate_model(best_model, X_train, y_train, label_names)
    val_results = evaluate_model(best_model, X_val, y_val, label_names)
    
    gap = train_results['accuracy'] - val_results['accuracy']
    
    model_file = MODEL_DIR / 'xgboost_regularized.pkl'
    joblib.dump(best_model, model_file)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, val_results, train_time, feature_importance, gap

def train_lightgbm_optimized(X_train, y_train, X_val, y_val, label_names):
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
    
    start_time = time.time()
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    train_time = time.time() - start_time
    
    train_results = evaluate_model(model, X_train, y_train, label_names)
    val_results = evaluate_model(model, X_val, y_val, label_names)
    
    gap = train_results['accuracy'] - val_results['accuracy']
    
    model_file = MODEL_DIR / 'lightgbm_optimized.pkl'
    joblib.dump(model, model_file)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, val_results, train_time, feature_importance, gap

def train_ensemble(models, X_train, y_train, X_val, y_val, label_names):
    weights = []
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        f1 = models[name]['results']['f1_score']
        weights.append(f1)
    
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']['model']),
            ('xgb', models['XGBoost']['model']),
            ('lgb', models['LightGBM']['model'])
        ],
        voting='soft',
        weights=weights,
        n_jobs=4
    )
    
    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    train_results = evaluate_model(voting_clf, X_train, y_train, label_names)
    val_results = evaluate_model(voting_clf, X_val, y_val, label_names)
    
    gap = train_results['accuracy'] - val_results['accuracy']
    
    model_file = MODEL_DIR / 'ensemble_weighted.pkl'
    joblib.dump(voting_clf, model_file)
    
    return voting_clf, val_results, train_time, gap

def compare_models(results_dict):
    comparison = []
    for model_name, data in results_dict.items():
        comparison.append({
            'Model': model_name,
            'Val Accuracy': data['results']['accuracy'],
            'Val F1-Score': data['results']['f1_score'],
            'Overfitting Gap': data['gap'],
            'Training Time (min)': data['train_time'] / 60
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('Val F1-Score', ascending=False)
    
    df_comparison['score'] = df_comparison['Val F1-Score'] - (df_comparison['Overfitting Gap'] * 0.5)
    best_idx = df_comparison['score'].idxmax()
    best_model = df_comparison.loc[best_idx, 'Model']
    
    comparison_file = RESULTS_DIR / 'model_comparison_regularized.csv'
    df_comparison.to_csv(comparison_file, index=False)
    
    return df_comparison, best_model

def main():
    X_train, y_train, X_val, y_val, feature_list, label_names = load_data()
    
    results_dict = {}
    
    rf_model, rf_results, rf_time, rf_importance, rf_gap = train_random_forest_regularized(
        X_train, y_train, X_val, y_val, label_names
    )
    results_dict['Random Forest'] = {
        'model': rf_model,
        'results': rf_results,
        'train_time': rf_time,
        'feature_importance': rf_importance,
        'gap': rf_gap
    }
    
    xgb_model, xgb_results, xgb_time, xgb_importance, xgb_gap = train_xgboost_regularized(
        X_train, y_train, X_val, y_val, label_names
    )
    results_dict['XGBoost'] = {
        'model': xgb_model,
        'results': xgb_results,
        'train_time': xgb_time,
        'feature_importance': xgb_importance,
        'gap': xgb_gap
    }
    
    lgb_model, lgb_results, lgb_time, lgb_importance, lgb_gap = train_lightgbm_optimized(
        X_train, y_train, X_val, y_val, label_names
    )
    results_dict['LightGBM'] = {
        'model': lgb_model,
        'results': lgb_results,
        'train_time': lgb_time,
        'feature_importance': lgb_importance,
        'gap': lgb_gap
    }
    
    ensemble_model, ensemble_results, ensemble_time, ensemble_gap = train_ensemble(
        results_dict, X_train, y_train, X_val, y_val, label_names
    )
    results_dict['Ensemble'] = {
        'model': ensemble_model,
        'results': ensemble_results,
        'train_time': ensemble_time,
        'feature_importance': None,
        'gap': ensemble_gap
    }
    
    df_comparison, best_model_name = compare_models(results_dict)

if __name__ == "__main__":
    main()