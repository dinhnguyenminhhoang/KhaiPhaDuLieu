import json
import pandas as pd
import numpy as np
import random
import Levenshtein
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

COMMON_SUBSTITUTIONS = {
    'ei': 'ie', 'ie': 'ei',        # Nhầm lẫn ei/ie: receive → recieve, believe → beleive
    'a': 'e', 'e': 'a',            # Nhầm nguyên âm a/e: cat → cet, bed → bad
    'i': 'y', 'y': 'i',            # Nhầm i/y: family → familiy, happy → happi
    'c': 'k', 'k': 'c',            # Nhầm c/k: cake → cace, pack → pack
    's': 'c', 'c': 's',            # Nhầm s/c: science → ssience, city → sity
    'f': 'ph', 'ph': 'f',          # Nhầm f/ph: phone → fone, fun → phun
    'tion': 'sion', 'sion': 'tion', # Nhầm đuôi tion/sion: action → acsion, vision → vition
} # các lỗi phổ biến hay gặp

KEYBOARD_NEIGHBORS = {
    'a': ['s', 'q', 'w', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
} # các lỗi do nhầm lẫn bàn phím

def load_dictionary_and_frequency():
    print("\n" + "="*60)
    print("BƯỚC 2: TẠO  DATASET")
    print("="*60)
    
    print("\n[2.1] Đang load dữ liệu...")
    
    with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
        dictionary = set(f.read().splitlines())
    print(f"  Dictionary: {len(dictionary):,} từ")
    
    with open(WORD_FREQ_FILE, 'r', encoding='utf-8') as f:
        word_freq = json.load(f)
    print(f"  Word frequencies: {len(word_freq):,} từ")
    
    return dictionary, word_freq

def select_candidate_words(word_freq):
    print("\n[2.2] Chọn từ ứng cử...")
    #lọc theo đọ dài
    candidates = []
    for word, freq in word_freq.items():
        if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH:
            candidates.append((word, freq))
    # Sắp xếp theo tần suất giảm dần
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Lấy top 100,000 từ phổ biến
    top_candidates = [w for w, f in candidates[:100000]]
    
    print(f"  Đã chọn {len(top_candidates):,} từ ứng cử")
    
    return top_candidates, {w: f for w, f in candidates}

def generate_insertion_error(word): #Tạo lỗi thêm
    if len(word) < 3:
        return None
    # cọn vị trí ngẫu hiên để chè kí tự 1-> hết
    pos = random.randint(1, len(word))
    
    if random.random() < 0.8 and pos < len(word):
        char = word[pos] if pos < len(word) else word[pos-1]
    else:
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
    # Thêm kí tự vào vị trí ngẫu nhiên
    return word[:pos] + char + word[pos:]

def generate_deletion_error(word): #Tạo lỗi xóa
    if len(word) < 4:
        return None
    # chọn vị trí ngẫu nhiên để xóa
    vowel_positions = [i for i, c in enumerate(word) if c in 'aeiou']
    # ưu tiên xóa nguyên âm
    if vowel_positions and random.random() < 0.6:
        pos = random.choice(vowel_positions)
    else:
        pos = random.randint(1, len(word)-2)
    
    return word[:pos] + word[pos+1:]

def generate_substitution_error(word): #Tạo lỗi thay thế
    if len(word) < 3:
        return None
    
    pos = random.randint(1, len(word)-1)
    old_char = word[pos].lower()
    # ưu tiên thay thế theo bàn phím
    if old_char in KEYBOARD_NEIGHBORS and random.random() < 0.4:
        new_char = random.choice(KEYBOARD_NEIGHBORS[old_char])
    # ưu tiên thay thế theo danh sách phổ biến
    elif old_char in COMMON_SUBSTITUTIONS and random.random() < 0.4:
        new_char = COMMON_SUBSTITUTIONS[old_char]
    else:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        new_char = random.choice([c for c in alphabet if c != old_char])
    
    return word[:pos] + new_char + word[pos+1:]

def generate_transposition_error(word): #Tạo lỗi hoán đổi
    if len(word) < 3:
        return None
    # chọn vị trí ngẫu nhiên để hoán đổi
    pos = random.randint(0, len(word)-2)
    chars = list(word)
    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    return ''.join(chars)

def create_error_dataset(candidates, word_freq_dict, dictionary, num_samples):
    print(f"\n[2.3] Đang tạo {num_samples:,} mẫu lỗi...")
    # tạo dataset
    dataset = []
    error_generators = {
        'insertion': generate_insertion_error,
        'deletion': generate_deletion_error,
        'substitution': generate_substitution_error,
        'transposition': generate_transposition_error,
    }
    
    error_distribution = {
        'insertion': 0.25,
        'deletion': 0.25,
        'substitution': 0.25,
        'transposition': 0.25,
    }
    
    attempts = 0
    max_attempts = num_samples * 3
    
    with tqdm(total=num_samples, desc="Tạo lỗi", unit="sample") as pbar:
        while len(dataset) < num_samples and attempts < max_attempts:
            attempts += 1
            
            correct_word = random.choice(candidates)
            
            error_type = random.choices(
                list(error_distribution.keys()),
                weights=list(error_distribution.values())
            )[0]
            
            generator = error_generators[error_type]
            incorrect_word = generator(correct_word)
            
            if incorrect_word and incorrect_word != correct_word:
                if incorrect_word not in dictionary:
                    edit_dist = Levenshtein.distance(correct_word, incorrect_word)
                    
                    if edit_dist == 1: 
                        dataset.append({
                            'id': len(dataset) + 1,
                            'correct_word': correct_word,
                            'incorrect_word': incorrect_word,
                            'error_type': error_type,
                            'edit_distance': edit_dist,
                            'word_length': len(correct_word),
                            'word_frequency': word_freq_dict.get(correct_word, 0),
                        })
                        pbar.update(1)
    
    print(f"  Đã tạo {len(dataset):,} mẫu lỗi")
    return dataset

def add_correct_samples(candidates, word_freq_dict, num_samples):
    print(f"\n[2.4] Thêm {num_samples:,} mẫu đúng...")
    
    correct_samples = []
    selected = random.sample(candidates, min(num_samples, len(candidates)))
    
    for correct_word in tqdm(selected, desc="Tạo mẫu đúng"):
        correct_samples.append({
            'id': 0,
            'correct_word': correct_word,
            'incorrect_word': correct_word,
            'error_type': 'correct',
            'edit_distance': 0,
            'word_length': len(correct_word),
            'word_frequency': word_freq_dict.get(correct_word, 0),
        })
    
    print(f"  Đã thêm {len(correct_samples):,} mẫu đúng")
    return correct_samples

def create_final_dataset(error_data, correct_data):
    print("\n[2.5] Tạo dataset cuối cùng...")
    
    all_data = error_data + correct_data
    df = pd.DataFrame(all_data)
    
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df['id'] = range(1, len(df) + 1)
    
    print(f"\n  THỐNG KÊ DATASET:")
    print(f"  - Tổng số mẫu: {len(df):,}")
    print(f"  - Số thuộc tính: {len(df.columns)}")
    print(f"\n  Phân bố labels:")
    
    label_counts = df['error_type'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"    {label:>15}: {count:>7,} ({pct:>5.2f}%)")
    
    return df

def save_dataset(df):
    print("\n[2.6] Lưu dataset...")
    
    df.to_csv(DATASET_FILE, index=False, encoding='utf-8')
    print(f"  Saved: {DATASET_FILE}")
    print(f"    Size: {len(df):,} rows × {len(df.columns)} columns")
    
    sample_file = PROCESSED_DATA_DIR / 'dataset_sample.csv'
    df.head(100).to_csv(sample_file, index=False, encoding='utf-8')
    print(f"  Saved sample: {sample_file}")
    
    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 2!")
    print(f"{'='*60}")
    print(f"\nDataset đã được tạo thành công:")
    print(f"  - File: {DATASET_FILE}")
    print(f"  - Tổng mẫu: {len(df):,}")
    print(f"  - Target accuracy: 75-85%")

def main():
    dictionary, word_freq = load_dictionary_and_frequency()
    candidates, word_freq_dict = select_candidate_words(word_freq)
    
    error_data = create_error_dataset(
        candidates, word_freq_dict, dictionary, NUM_SAMPLES
    )
    
    correct_data = add_correct_samples(
        candidates, word_freq_dict, NUM_CORRECT_SAMPLES
    )
    
    df = create_final_dataset(error_data, correct_data)
    save_dataset(df)

if __name__ == "__main__":
    main()