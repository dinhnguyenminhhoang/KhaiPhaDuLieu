import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

sns.set_style("whitegrid")

def load_dataset():
    print("\n" + "="*60)
    print("BƯỚC 3: TIỀN XỬ LÝ")
    print("="*60)
    
    print("\n[3.1] Đang load dataset...")
    
    df = pd.read_csv(DATASET_FILE)
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    return df

def check_data_quality(df):
    print("\n[3.2] Kiểm tra chất lượng...")
    
    missing = df.isnull().sum()
    print(f"  Missing values: {missing.sum()}")
    
    if missing.sum() > 0:
        print(f"  Phát hiện missing values, đang xử lý...")
        df = df.dropna()
        print(f"  Dataset sau xử lý: {len(df):,} rows")
    
    dup_count = df.duplicated().sum()
    print(f"  Duplicates: {dup_count}")
    
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"  Đã loại bỏ duplicates")
    
    return df

def feature_engineering(df):
    print("\n[3.3] Feature Engineering...")
    
    df_processed = df.copy()
    
    source_col = 'incorrect_word'
    
    print("  [3.3.1] Character features...")
    
    def count_vowels(word):
        return sum(1 for c in str(word).lower() if c in 'aeiou')
    
    def count_consonants(word):
        return len(str(word)) - count_vowels(word)
    # Số nguyên âm
    df_processed['num_vowels'] = df_processed[source_col].apply(count_vowels)
    # Số phụ âm
    df_processed['num_consonants'] = df_processed[source_col].apply(count_consonants)
    # Tỷ lệ nguyên âm
    df_processed['vowel_ratio'] = df_processed['num_vowels'] / df_processed['word_length']
    # Tỷ lệ phụ âm
    df_processed['consonant_ratio'] = df_processed['num_consonants'] / df_processed['word_length']
    
    print("  [3.3.2] N-gram features...")
    # Chữ đầu tiên
    df_processed['first_char'] = df_processed[source_col].str[0]
    # Chữ cuối cùng
    df_processed['last_char'] = df_processed[source_col].str[-1]
    # Bigram đầu tiên
    df_processed['first_bigram'] = df_processed[source_col].str[:2]
    # Bigram cuối cùng
    df_processed['last_bigram'] = df_processed[source_col].str[-2:]
    # Trigram đầu tiên
    df_processed['first_trigram'] = df_processed[source_col].str[:3]
    # Trigram cuối cùng
    df_processed['last_trigram'] = df_processed[source_col].str[-3:]
    
    # Bigram phổ biến
    df_processed['first_bigram_common'] = df_processed['first_bigram'].apply(
        lambda x: 1 if x in ['th', 'in', 'an', 'er', 'on', 're', 'st', 'en'] else 0
    )
    # Bigram phổ biến
    df_processed['last_bigram_common'] = df_processed['last_bigram'].apply(
        lambda x: 1 if x in ['ed', 'er', 'ly', 'ng', 'al', 'on', 'es', 'en'] else 0
    )
    
    print("  [3.3.3] Pattern features...")
    # Có hai chữ trùng
    def has_double_letters(word):
        word = str(word)
        for i in range(len(word)-1):
            if word[i] == word[i+1]:
                return 1
        return 0
    
    # Số chữ trùng
    def count_double_letters(word):
        word = str(word)
        count = 0
        for i in range(len(word)-1):
            if word[i] == word[i+1]:
                count += 1
        return count
    # Có hai nguyên âm trùng
    df_processed['has_double_letters'] = df_processed[source_col].apply(has_double_letters)
    # Số nguyên âm trùng
    df_processed['num_double_letters'] = df_processed[source_col].apply(count_double_letters)
    
    def has_repeated_vowels(word):
        word = str(word)
        vowels = [c for c in word.lower() if c in 'aeiou']
        for i in range(len(vowels)-1):
            if vowels[i] == vowels[i+1]:
                return 1
        return 0
    # Có hai nguyên âm trùng
    df_processed['has_repeated_vowels'] = df_processed[source_col].apply(has_repeated_vowels)
    
    def is_alternating(word):
        word = str(word).lower()
        if len(word) < 3:
            return 0
        
        pattern = []
        for char in word:
            if char in 'aeiou':
                pattern.append('V')
            else:
                pattern.append('C')
        
        for i in range(len(pattern)-1):
            if pattern[i] == pattern[i+1]:
                return 0
        return 1
    
    df_processed['is_alternating'] = df_processed[source_col].apply(is_alternating)
    
    print("  [3.3.4] Complexity features...")
    
    def count_syllables(word):
        word = str(word).lower()
        count = 0
        vowels = 'aeiou'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        
        return count
    
    df_processed['syllable_count'] = df_processed[source_col].apply(count_syllables)
    df_processed['syllable_ratio'] = df_processed['syllable_count'] / df_processed['word_length']
    
    def max_consonant_cluster(word):
        word = str(word)
        max_len = 0
        current_len = 0
        for char in word.lower():
            if char not in 'aeiou':
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len
    
    def max_vowel_cluster(word):
        word = str(word)
        max_len = 0
        current_len = 0
        for char in word.lower():
            if char in 'aeiou':
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len
    
    df_processed['max_consonant_cluster'] = df_processed[source_col].apply(max_consonant_cluster)
    df_processed['max_vowel_cluster'] = df_processed[source_col].apply(max_vowel_cluster)
    
    print("  [3.3.5] Structural features...")
    
    def char_diversity(word):
        if len(word) == 0:
            return 0
        return len(set(word)) / len(word)
    
    df_processed['char_diversity'] = df_processed[source_col].apply(char_diversity)
    
    def unique_vowels(word):
        return len(set(c for c in str(word).lower() if c in 'aeiou'))
    
    def unique_consonants(word):
        return len(set(c for c in str(word).lower() if c not in 'aeiou'))
    
    df_processed['unique_vowels'] = df_processed[source_col].apply(unique_vowels)
    df_processed['unique_consonants'] = df_processed[source_col].apply(unique_consonants)
    
    print("  [3.3.6] Position features...")
    
    df_processed['starts_with_vowel'] = df_processed[source_col].str[0].str.lower().isin(list('aeiou')).astype(int)
    df_processed['ends_with_vowel'] = df_processed[source_col].str[-1].str.lower().isin(list('aeiou')).astype(int)
    
    def get_middle_char(word):
        if len(word) == 0:
            return 'x'
        mid = len(word) // 2
        return word[mid]
    
    df_processed['middle_char'] = df_processed[source_col].apply(get_middle_char)
    df_processed['middle_is_vowel'] = df_processed['middle_char'].str.lower().isin(list('aeiou')).astype(int)
    
    new_features = [col for col in df_processed.columns if col not in df.columns]
    print(f"\n  Đã tạo {len(new_features)} features mới")
    print(f"  Tổng features: {len(df_processed.columns)}")
    
    print(f"\n  FEATURES MỚI:")
    for feat in sorted(new_features):
        print(f"    - {feat}")
    
    return df_processed

def encode_labels(df):
    print("\n[3.4] Encoding labels...")
    
    df_encoded = df.copy()
    
    le_error = LabelEncoder()
    df_encoded['error_type_encoded'] = le_error.fit_transform(df_encoded['error_type'])
    
    print(f"  Encoded error_type:")
    for i, label in enumerate(le_error.classes_):
        count = (df_encoded['error_type_encoded'] == i).sum()
        print(f"    {label:>15} → {i} ({count:>7,} samples)")
    
    categorical_cols = ['first_char', 'last_char', 'first_bigram', 'last_bigram', 
                       'first_trigram', 'last_trigram', 'middle_char']
    
    encoders_dict = {'error_type': {label: i for i, label in enumerate(le_error.classes_)}}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
            encoders_dict[col] = {label: i for i, label in enumerate(le.classes_)}
    
    encoder_file = PROCESSED_DATA_DIR / 'label_encoders.json'
    with open(encoder_file, 'w', encoding='utf-8') as f:
        json.dump(encoders_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved encoders: {encoder_file}")
    
    return df_encoded, encoders_dict

def select_features_for_training(df):
    print("\n[3.5] Chọn features cho training...")
    
    numerical_features = [
        'word_length',
        'num_vowels',
        'num_consonants',
        'vowel_ratio',
        'consonant_ratio',
        'first_char_encoded',
        'last_char_encoded',
        'first_bigram_encoded',
        'last_bigram_encoded',
        'first_trigram_encoded',
        'last_trigram_encoded',
        'first_bigram_common',
        'last_bigram_common',
        'has_double_letters',
        'num_double_letters',
        'has_repeated_vowels',
        'is_alternating',
        'syllable_count',
        'syllable_ratio',
        'max_consonant_cluster',
        'max_vowel_cluster',
        'char_diversity',
        'unique_vowels',
        'unique_consonants',
        'starts_with_vowel',
        'ends_with_vowel',
        'middle_char_encoded',
        'middle_is_vowel',
    ]
    
    target = 'error_type_encoded'
    
    available_features = [f for f in numerical_features if f in df.columns]
    
    print(f"  Selected {len(available_features)} features")
    print(f"\n  FEATURE LIST:")
    for i, feat in enumerate(available_features, 1):
        print(f"    {i:2d}. {feat}")
    
    return available_features, target
# Chuẩn hóa features về 1 thang đo (0-1) để dễ dàng so sánh
def normalize_features(df, features):
    print("\n[3.6] Chuẩn hóa features...")
    
    df_normalized = df.copy()
    
    scaler = StandardScaler()
    df_normalized[features] = scaler.fit_transform(df[features])
    
    print(f"  Normalized {len(features)} features")
    
    import joblib
    scaler_file = MODEL_DIR / 'feature_scaler.pkl'
    joblib.dump(scaler, scaler_file)
    print(f"  Saved scaler: {scaler_file}")
    
    return df_normalized, scaler

def split_dataset(df, features, target):
    print("\n[3.7] Chia dataset...")
    
    from sklearn.model_selection import GroupShuffleSplit
    
    X = df[features]
    y = df[target]
    groups = df['correct_word']
    
    test_val_size = VAL_SIZE + TEST_SIZE
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_val_size, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_temp = groups.iloc[temp_idx]
    
    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1-val_ratio, random_state=RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    
    X_val = X_temp.iloc[val_idx]
    y_val = y_temp.iloc[val_idx]
    X_test = X_temp.iloc[test_idx]
    y_test = y_temp.iloc[test_idx]
    
    print(f"  Train: {len(X_train):>7,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(X_val):>7,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(X_test):>7,} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(df, X_train, X_val, X_test, y_train, y_val, y_test, features):
    print("\n[3.8] Lưu splits...")
    
    train_df = df.loc[X_train.index].copy()
    val_df = df.loc[X_val.index].copy()
    test_df = df.loc[X_test.index].copy()
    
    train_df.to_csv(TRAIN_FILE, index=False, encoding='utf-8')
    val_df.to_csv(VAL_FILE, index=False, encoding='utf-8')
    test_df.to_csv(TEST_FILE, index=False, encoding='utf-8')
    
    print(f"  Saved: {TRAIN_FILE}")
    print(f"  Saved: {VAL_FILE}")
    print(f"  Saved: {TEST_FILE}")
    
    feature_file = PROCESSED_DATA_DIR / 'feature_list.json'
    with open(feature_file, 'w', encoding='utf-8') as f:
        json.dump({'features': features}, f, indent=2)
    print(f"  Saved feature list: {feature_file}")

def main():
    df = load_dataset()
    df = check_data_quality(df)
    df = feature_engineering(df)
    df, encoders = encode_labels(df)
    features, target = select_features_for_training(df)
    df, scaler = normalize_features(df, features)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, features, target)
    save_splits(df, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 3!")
    print(f"{'='*60}")
    print(f"\nDữ liệu đã sẵn sàng với {len(features)} features!")
    print(f"Tiếp theo: Chạy BƯỚC 4 để train models")

if __name__ == "__main__":
    main()