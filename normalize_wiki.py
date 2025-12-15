import os
import re
import json
from pathlib import Path
from collections import Counter
from lxml import etree as ET

XML_FILE = r"C:/saveTempData/spell_check/data/enwiki-latest-pages-articles.xml"  # file XML ~100GB
OUTPUT_DIR = "processed_data"
MAX_ARTICLES = 200_000   

def clean_text(text: str) -> str:
    """Làm sạch văn bản Wikipedia cơ bản"""
    if not text:
        return ""
    # bỏ template {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    # link nội bộ [[a|b]] -> b ; [[a]] -> a
    text = re.sub(r"\[\[.*?\|(.+?)\]\]", r"\1", text)
    text = re.sub(r"\[\[(.+?)\]\]", r"\1", text)
    # bỏ heading == ==
    text = re.sub(r"==+.*?==+", " ", text)
    # bỏ HTML tag
    text = re.sub(r"<[^>]+>", " ", text)
    # bỏ URL
    text = re.sub(r"http\S+", " ", text)
    # bỏ ký tự lạ, chỉ giữ chữ, số, dấu cơ bản
    text = re.sub(r"[^\w\s.,!?;:'-]", " ", text)
    # gộp space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_words(text: str):
    """Tách từ (tiếng Anh ≥2 chữ cái)"""
    text = text.lower()
    return re.findall(r"\b[a-z]{2,}\b", text)

def save_intermediate(word_counter: Counter, out_dir: Path, article_count: int):
    """Lưu kết quả tạm"""
    filtered = {w: f for w, f in word_counter.items() if f >= 5}
    out = out_dir / f"wiki_words_intermediate_{article_count}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"  Saved intermediate at {out} ({len(filtered)} words)")

def save_final(word_counter: Counter, out_dir: Path):
    """Lưu kết quả cuối"""
    out_dir.mkdir(exist_ok=True)
    print("Saving final results...")

    for min_freq in [5, 10, 20, 50]:
        filtered = {w: f for w, f in word_counter.items() if f >= min_freq}
        out = out_dir / f"wiki_words_freq{min_freq}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(filtered)} words (freq>={min_freq}) -> {out}")

    # top dictionary
    top_words = [w for w, _ in word_counter.most_common(200_000)]
    with open(out_dir / "wiki_dictionary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(top_words))
    print(f"Saved top {len(top_words)} words -> wiki_dictionary.txt")

    stats = {
        "total_unique_words": len(word_counter),
        "total_word_occurrences": sum(word_counter.values()),
        "most_common_10": word_counter.most_common(10),
    }
    with open(out_dir / "wiki_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Saved stats -> wiki_stats.json")

def process_wiki(xml_file: str, max_articles=None, out_dir="processed_data"):
    print(f"Processing {xml_file} ...")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    counter = Counter()
    article_count = 0

    context = ET.iterparse(xml_file, events=("end",), tag="{*}page", huge_tree=True, recover=True)

    for _, elem in context:
        title_el = elem.find(".//{*}title")
        text_el = elem.find(".//{*}text")

        title = title_el.text if title_el is not None else None
        text = text_el.text if text_el is not None else None

        if title and text:
            if not text.startswith("#REDIRECT") and not any(
                title.startswith(ns) for ns in ("File:", "Category:", "Template:", "Wikipedia:", "Help:", "MediaWiki:")
            ):
                clean = clean_text(text)
                if len(clean) > 100:
                    words = extract_words(clean)
                    if len(words) > 20:
                        counter.update(words)
                        article_count += 1

                        if article_count % 10_000 == 0:
                            print(f"Processed {article_count} articles, {len(counter)} unique words")
                        if article_count % 50_000 == 0:
                            save_intermediate(counter, out_dir, article_count)

                        if max_articles and article_count >= max_articles:
                            print(f"Reached limit {max_articles}")
                            break

        # giải phóng memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print(f"Done. Total articles: {article_count}, unique words: {len(counter)}")
    return counter

if __name__ == "__main__":
    if not os.path.exists(XML_FILE):
        print(f"File not found: {XML_FILE}")
    else:
        word_counter = process_wiki(XML_FILE, MAX_ARTICLES, OUTPUT_DIR)
        save_final(word_counter, Path(OUTPUT_DIR))
