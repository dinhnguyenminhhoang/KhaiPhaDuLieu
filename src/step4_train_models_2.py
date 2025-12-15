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
    print("\n" + "="*60)
    print("BƯỚC 4: TRAIN MODELS")
    print("="*60)
    
    print("\n[4.1] Đang load dữ liệu...")
    
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    print(f"  Train set: {len(train_df):,} samples")
    print(f"  Val set: {len(val_df):,} samples")
    
    with open(PROCESSED_DATA_DIR / 'feature_list.json', 'r') as f:
        feature_list = json.load(f)['features']
    
    print(f"  Features: {len(feature_list)}")
    
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
# model random_forest_regularized
def train_random_forest_regularized(X_train, y_train, X_val, y_val, label_names):
    print("\n[4.2] Training Random Forest...")
    
    param_distributions = {
        'n_estimators': [300, 500, 700], # số lượng cây
        'max_depth': [25, 30, 35], # độ sâu tối đa của cây
        'min_samples_split': [2, 3, 5], # số lượng mẫu tối thiểu để chia
        'min_samples_leaf': [1, 2, 3], # số lượng mẫu tối thiểu ở lá
        'max_features': ['sqrt', 'log2'], # số lượng đặc trưng tối đa để chọn
        'min_impurity_decrease': [0.0, 0.0001, 0.001], # giảm thiểu thông tin tối thiểu
        'bootstrap': [True],# sử dụng bootstrap(lấy ngẫu nhiên có hoàn lại->tạo dữ liệu con khác nhau cho mỗi cây)
        'class_weight': ['balanced', 'balanced_subsample']# cân bằng
    }
    # tạo model
    base_rf = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=4,# số lượng thread(4 CPU cores)
        verbose=0 # không in thông tin chi tiết
    )
    
    print("  Random Search (15 iterations, 3-fold CV)...")
    random_search = RandomizedSearchCV(
        estimator=base_rf,#model cơ sở 
        param_distributions=param_distributions,
        n_iter=15, # số lần thử
        cv=3, # số fold
        scoring='f1_weighted',# scoring
        n_jobs=2,
        random_state=RANDOM_SEED,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"\n  Time: {train_time/60:.2f} min")
    print(f"\n  BEST PARAMS:")
    for key, value in random_search.best_params_.items():
        print(f"     {key}: {value}")
    
    best_model = random_search.best_estimator_
    
    train_results = evaluate_model(best_model, X_train, y_train, label_names)
    val_results = evaluate_model(best_model, X_val, y_val, label_names)
    
    print(f"\n  RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")
    
    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")
    
    model_file = MODEL_DIR / 'random_forest.pkl'
    joblib.dump(best_model, model_file)
    print(f"\n  Saved: {model_file}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, val_results, train_time, feature_importance, gap

def train_xgboost_regularized(X_train, y_train, X_val, y_val, label_names):
    print("\n[4.3] Training XGBoost...")
    
    param_grid = {
        'n_estimators': [500, 700, 1000], # số lượng cây
        'max_depth': [8, 10, 12], # độ sâu tối đa của cây
        'learning_rate': [0.05, 0.07, 0.1], # tốc độ học
        'subsample': [0.8, 0.9], # tỷ lệ mẫu để huấn luyện
        'colsample_bytree': [0.8, 0.9], # tỷ lệ đặc trưng để huấn luyện
        'min_child_weight': [1, 3, 5], # số lượng mẫu tối thiểu ở lá
        'gamma': [0, 0.1, 0.2], # giảm thiểu thông tin tối thiểu
        'reg_alpha': [0, 0.01, 0.1],# L1 regularization
        'reg_lambda': [1, 1.5, 2]# L2 regularization
    }
    
    base_xgb = xgb.XGBClassifier(
        tree_method='hist',# sử dụng histogram -EXACT(data nhỏ) -GPU_HIST(data lớn)
        random_state=RANDOM_SEED, 
        n_jobs=4,
        eval_metric='mlogloss' # metric
    )
    
    print("  Random Search (20 iterations, 3-fold CV)...")
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
    
    print(f"\n  Time: {train_time/60:.2f} min")
    print(f"\n  BEST PARAMS:")
    for key, value in random_search.best_params_.items():
        print(f"     {key}: {value}")
    
    best_model = random_search.best_estimator_
    
    train_results = evaluate_model(best_model, X_train, y_train, label_names)
    val_results = evaluate_model(best_model, X_val, y_val, label_names)
    
    print(f"\n  RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")
    
    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")
    
    model_file = MODEL_DIR / 'xgboost.pkl'
    joblib.dump(best_model, model_file)
    print(f"\n  Saved: {model_file}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, val_results, train_time, feature_importance, gap

def train_lightgbm_optimized(X_train, y_train, X_val, y_val, label_names):
    print("\n[4.4] Training LightGBM...")
    
    params = {
        'n_estimators': 1000, # số lượng cây
        'max_depth': 15, # độ sâu tối đa của cây
        'learning_rate': 0.05, # tốc độ học
        'num_leaves': 63, # số lượng lá
        'subsample': 0.8, # tỷ lệ mẫu
        'colsample_bytree': 0.8, # tỷ lệ đặc trưng
        'min_child_samples': 20, # số lượng mẫu tối thiểu ở lá
        'reg_alpha': 0.1, # L1 regularization
        'reg_lambda': 0.1, # L2 regularization
        'feature_fraction': 0.8, # tỷ lệ đặc trưng
        'bagging_fraction': 0.8, # tỷ lệ mẫu
        'bagging_freq': 5, # số lần lặp
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
    print(f"\n  Time: {train_time:.2f}s")
    
    train_results = evaluate_model(model, X_train, y_train, label_names)
    val_results = evaluate_model(model, X_val, y_val, label_names)
    
    print(f"\n  RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")
    
    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")
    
    model_file = MODEL_DIR / 'lightgbm.pkl'
    joblib.dump(model, model_file)
    print(f"\n  Saved: {model_file}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, val_results, train_time, feature_importance, gap

# ENSEMBLE LEARNING: Kết hợp nhiều models để tạo ra model mạnh hơn
def train_ensemble(models, X_train, y_train, X_val, y_val, label_names):
    print("\n[4.5] Training Ensemble (Weighted Voting)...")
    
    # BƯỚC 1: Tính trọng số dựa trên F1-Score của từng model
    # Model nào F1 cao hơn sẽ có ảnh hưởng lớn hơn đến kết quả cuối cùng
    weights = []
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        f1 = models[name]['results']['f1_score']
        weights.append(f1)
    
    # Chuẩn hóa weights về tổng = 1.0 (để dễ hiểu phần trăm đóng góp)
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    print(f"\n  WEIGHTS:")
    for name, weight in zip(['Random Forest', 'XGBoost', 'LightGBM'], weights):
        print(f"     {name}: {weight:.3f}")
    
    # BƯỚC 2: Tạo VotingClassifier
    voting_clf = VotingClassifier(
        # estimators: danh sách các models đã train (tên, model_object)
        estimators=[
            ('rf', models['Random Forest']['model']),
            ('xgb', models['XGBoost']['model']),
            ('lgb', models['LightGBM']['model'])
        ],
        # voting='soft': dùng xác suất thay vì chỉ đếm class (chính xác hơn)
        # Công thức: P_final = (w1*P1 + w2*P2 + w3*P3) / (w1+w2+w3)
        voting='soft',
        # weights: trọng số cho từng model (model tốt hơn có weight cao hơn)
        weights=weights,
        # n_jobs: số CPU cores để predict song song
        n_jobs=4
    )
    
    # BƯỚC 3: "Train" ensemble (thực chất chỉ setup, không train lại models)
    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"\n  Time: {train_time:.2f}s")
    
    # BƯỚC 4: Đánh giá ensemble trên train và validation set
    train_results = evaluate_model(voting_clf, X_train, y_train, label_names)
    val_results = evaluate_model(voting_clf, X_val, y_val, label_names)
    
    print(f"\n  RESULTS:")
    print(f"    TRAIN: Acc={train_results['accuracy']:.4f} | F1={train_results['f1_score']:.4f}")
    print(f"    VAL:   Acc={val_results['accuracy']:.4f} | F1={val_results['f1_score']:.4f}")
    
    gap = train_results['accuracy'] - val_results['accuracy']
    print(f"    GAP:   {gap:.4f} ({gap*100:.2f}%)")
    
    # BƯỚC 5: So sánh ensemble với từng model riêng lẻ
    print(f"\n  COMPARISON:")
    print(f"    Random Forest Val F1: {models['Random Forest']['results']['f1_score']:.4f}")
    print(f"    XGBoost Val F1:       {models['XGBoost']['results']['f1_score']:.4f}")
    print(f"    LightGBM Val F1:      {models['LightGBM']['results']['f1_score']:.4f}")
    print(f"    Ensemble Val F1:      {val_results['f1_score']:.4f}")
    
    # Tính độ cải thiện so với model tốt nhất
    improvement = val_results['f1_score'] - max(
        models['Random Forest']['results']['f1_score'],
        models['XGBoost']['results']['f1_score'],
        models['LightGBM']['results']['f1_score']
    )
    
    # Ensemble tốt hơn hay tệ hơn model tốt nhất?
    if improvement > 0:
        print(f"    Ensemble BETTER by {improvement:.4f}")
    else:
        print(f"    Ensemble WORSE by {abs(improvement):.4f}")
    
    # BƯỚC 6: Lưu ensemble model
    model_file = MODEL_DIR / 'ensemble_weighted.pkl'
    joblib.dump(voting_clf, model_file)
    print(f"\n  Saved: {model_file}")
    
    return voting_clf, val_results, train_time, gap

def compare_models(results_dict):
    print("\n" + "="*60)
    print("SO SÁNH MODELS")
    print("="*60)
    
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
    
    print("\nBẢNG SO SÁNH:")
    print(df_comparison.to_string(index=False))
    
    df_comparison['score'] = df_comparison['Val F1-Score'] - (df_comparison['Overfitting Gap'] * 0.5)
    best_idx = df_comparison['score'].idxmax()
    best_model = df_comparison.loc[best_idx, 'Model']
    
    print(f"\nBEST MODEL: {best_model}")
    print(f"   Val F1: {df_comparison.loc[best_idx, 'Val F1-Score']:.4f}")
    print(f"   Gap: {df_comparison.loc[best_idx, 'Overfitting Gap']:.4f}")
    
    comparison_file = RESULTS_DIR / 'model_comparison.csv'
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\nSaved: {comparison_file}")
    
    return df_comparison, best_model

def main():
    X_train, y_train, X_val, y_val, feature_list, label_names = load_data()
    
    results_dict = {}
    
    print("\n" + "="*60)
    print("BẮT ĐẦU TRAINING")
    print("="*60)
    
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
    
    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 4!")
    print(f"{'='*60}")
    print(f"\nĐã train {len(results_dict)} models")
    print(f"Best model: {best_model_name}")

if __name__ == "__main__":
    main()