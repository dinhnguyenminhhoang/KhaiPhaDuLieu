import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data():
    print("\n" + "="*60)
    print("BƯỚC 1: PHÂN TÍCH DỮ LIỆU WIKIPEDIA")
    print("="*60)

    print("\n[1.1] Đang load dữ liệu...")

    with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
        dictionary = f.read().splitlines()
    print(f"  Dictionary: {len(dictionary):,} từ")

    with open(WORD_FREQ_FILE, 'r', encoding='utf-8') as f:
        word_freq = json.load(f)
    print(f"  Word frequencies: {len(word_freq):,} từ")

    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"  Stats loaded")

    return dictionary, word_freq, stats

def analyze_basic_stats(dictionary, word_freq, stats):
    print("\n[1.2] Thống kê tổng quan...")

    report = []
    report.append("="*60)
    report.append("THỐNG KÊ TỔNG QUAN DỮ LIỆU WIKIPEDIA")
    report.append("="*60)
    report.append(f"Tổng số từ unique: {stats['total_unique_words']:,}")
    report.append(f"Tổng số lần xuất hiện: {stats['total_word_occurrences']:,}")
    report.append(f"Số từ có tần suất ≥10: {len(word_freq):,}")
    report.append("")

    for line in report:
        print(line)

    return report

def analyze_word_length(dictionary, word_freq):
    print("\n[1.3] Phân tích độ dài từ...")

    word_lengths = [len(word) for word in dictionary]

    df = pd.DataFrame({
        'word': dictionary,
        'length': word_lengths,
        'frequency': [word_freq.get(word, 0) for word in dictionary]
    })

    length_stats = df['length'].describe()

    report = []
    report.append("="*60)
    report.append("PHÂN TÍCH ĐỘ DÀI TỪ")
    report.append("="*60)
    report.append(f"Độ dài trung bình (mean): {length_stats['mean']:.2f} ký tự")
    report.append(f"Độ dài trung vị (median): {length_stats['50%']:.0f} ký tự")
    report.append(f"Độ dài tối thiểu: {length_stats['min']:.0f} ký tự")
    report.append(f"Độ dài tối đa: {length_stats['max']:.0f} ký tự")
    report.append(f"Độ lệch chuẩn (std): {length_stats['std']:.2f}")
    report.append("")

    length_dist = df['length'].value_counts().sort_index()
    report.append("Phân bố theo độ dài:")
    for length, count in length_dist.head(15).items():
        pct = (count / len(df)) * 100
        report.append(f"  {length} ký tự: {count:>7,} từ ({pct:>5.2f}%)")
    report.append("")

    for line in report:
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(word_lengths, bins=range(2, max(word_lengths)+2), 
                 edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Độ dài từ (ký tự)', fontsize=12)
    axes[0].set_ylabel('Số lượng từ', fontsize=12)
    axes[0].set_title('Phân bố độ dài từ trong Dictionary', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].boxplot(word_lengths, vert=True)
    axes[1].set_ylabel('Độ dài từ (ký tự)', fontsize=12)
    axes[1].set_title('Boxplot độ dài từ', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'word_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'word_length_distribution.png'}")
    plt.close()
    
    return report, df

def analyze_frequency(df):
    print("\n[1.4] Phân tích tần suất...")
    
    df_freq = df[df['frequency'] > 0].copy()
    df_freq = df_freq.sort_values('frequency', ascending=False)
    
    report = []
    report.append("="*60)
    report.append("PHÂN TÍCH TẦN SUẤT XUẤT HIỆN")
    report.append("="*60)
    
    report.append("Top 20 từ phổ biến nhất:")
    for idx, row in df_freq.head(20).iterrows():
        report.append(f"  {row['word']:>15} : {row['frequency']:>10,} lần")
    report.append("")
    
    report.append("Top 20 từ hiếm nhất (trong số từ có tần suất ≥10):")
    for idx, row in df_freq.tail(20).iterrows():
        report.append(f"  {row['word']:>15} : {row['frequency']:>10,} lần")
    report.append("")
    
    freq_stats = df_freq['frequency'].describe()
    report.append("Thống kê tần suất:")
    report.append(f"  Trung bình: {freq_stats['mean']:,.2f} lần")
    report.append(f"  Trung vị: {freq_stats['50%']:,.0f} lần")
    report.append(f"  Tần suất cao nhất: {freq_stats['max']:,.0f} lần")
    report.append(f"  Tần suất thấp nhất: {freq_stats['min']:,.0f} lần")
    report.append("")
    
    for line in report:
        print(line)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    top_30 = df_freq.head(30)
    ax.barh(range(len(top_30)), top_30['frequency'].values, color='coral')
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30['word'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Tần suất xuất hiện', fontsize=12)
    ax.set_title('Top 30 từ phổ biến nhất trong Wikipedia', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_words_frequency.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'top_words_frequency.png'}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = [10, 50, 100, 500, 1000, 5000, 10000, 50000, df_freq['frequency'].max()]
    df_freq['freq_bin'] = pd.cut(df_freq['frequency'], bins=bins, labels=[
        '10-50', '50-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '>50K'
    ])
    
    freq_dist = df_freq['freq_bin'].value_counts().sort_index()
    ax.bar(range(len(freq_dist)), freq_dist.values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(freq_dist)))
    ax.set_xticklabels(freq_dist.index, rotation=45)
    ax.set_xlabel('Khoảng tần suất', fontsize=12)
    ax.set_ylabel('Số lượng từ', fontsize=12)
    ax.set_title('Phân bố từ theo khoảng tần suất', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'frequency_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'frequency_distribution.png'}")
    plt.close()
    
    return report

def analyze_first_letters(df):
    print("\n[1.5] Phân tích chữ cái đầu...")
    
    df['first_letter'] = df['word'].str[0]
    letter_counts = df['first_letter'].value_counts().sort_index()
    
    report = []
    report.append("="*60)
    report.append("PHÂN TÍCH CHỮ CÁI ĐẦU")
    report.append("="*60)
    for letter, count in letter_counts.items():
        pct = (count / len(df)) * 100
        report.append(f"  {letter}: {count:>8,} từ ({pct:>5.2f}%)")
    report.append("")
    
    for line in report:
        print(line)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(letter_counts.index, letter_counts.values, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Chữ cái đầu tiên', fontsize=12)
    ax.set_ylabel('Số lượng từ', fontsize=12)
    ax.set_title('Phân bố từ theo chữ cái đầu tiên', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'first_letter_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'first_letter_distribution.png'}")
    plt.close()
    
    return report

def save_report(all_reports):
    print("\n[1.6] Lưu báo cáo...")
    
    report_file = RESULTS_DIR / 'data_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        for report in all_reports:
            for line in report:
                f.write(line + '\n')
            f.write('\n')
    
    print(f"  Saved: {report_file}")
    print(f"\n{'='*60}")
    print("HOÀN THÀNH BƯỚC 1!")
    print(f"{'='*60}")
    print(f"\nKết quả được lưu tại:")
    print(f"  - Báo cáo: {report_file}")
    print(f"  - Biểu đồ: {FIGURES_DIR}/")
    print(f"\nTiếp theo: Chạy BƯỚC 2 để tạo dataset spell check")

def main():
    dictionary, word_freq, stats = load_data()
    
    all_reports = []
    
    report1 = analyze_basic_stats(dictionary, word_freq, stats)
    all_reports.append(report1)
    
    report2, df = analyze_word_length(dictionary, word_freq)
    all_reports.append(report2)
    
    report3 = analyze_frequency(df)
    all_reports.append(report3)
    
    report4 = analyze_first_letters(df)
    all_reports.append(report4)
    
    save_report(all_reports)

if __name__ == "__main__":
    main()