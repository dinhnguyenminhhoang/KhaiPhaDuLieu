"""
FastAPI Backend cho Spell Checker
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path
import json
import joblib
import Levenshtein
import pandas as pd
import numpy as np
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

app = FastAPI(
    title="AI Spell Checker API",
    description="Spell Checker với ML - 84% Accuracy, Low Overfitting (<5%)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class CheckWordRequest(BaseModel):
    word: str
    model_name: str = "LightGBM"

class CheckSentenceRequest(BaseModel):
    sentence: str
    model_name: str = "LightGBM"

class WordCorrection(BaseModel):
    word: str
    distance: int
    frequency: int
    score: float

class ErrorDetail(BaseModel):
    original: str
    correction: str
    error_type: str
    confidence: float
    distance: int
    position: int

class CheckWordResponse(BaseModel):
    is_correct: bool
    word: str
    corrections: List[WordCorrection]
    error_type: Optional[str] = None
    confidence: Optional[float] = None
    probability_distribution: Optional[Dict[str, float]] = None

class CheckSentenceResponse(BaseModel):
    original_sentence: str
    corrected_sentence: str
    has_errors: bool
    num_errors: int
    errors: List[ErrorDetail]

# ==================== SPELL CHECKER CLASS ====================

class SpellCheckerAPI:
    def __init__(self):
        print("Loading resources...")
        
        # Load dictionary
        with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
            self.dictionary = set(f.read().splitlines())
        
        # Load frequencies
        with open(WORD_FREQ_FILE, 'r', encoding='utf-8') as f:
            self.word_freq = json.load(f)
        
        # Load models (Anti-Overfitting Versions)
        self.models = {}
        model_files = {
            'LightGBM': MODEL_DIR / 'lightgbm.pkl',
            'XGBoost': MODEL_DIR / 'xgboost.pkl',
            'RandomForest': MODEL_DIR / 'random_forest.pkl',
            'Ensemble': MODEL_DIR / 'ensemble_weighted.pkl'
        }
        
        for name, path in model_files.items():
            if path.exists():
                self.models[name] = joblib.load(path)
                print(f"  ✓ Loaded: {name}")
        
        # Load encoders
        with open(PROCESSED_DATA_DIR / 'label_encoders.json', 'r') as f:
            encoders = json.load(f)
        self.label_names = {v: k for k, v in encoders['error_type'].items()}
        
        # Load scaler
        self.scaler = joblib.load(MODEL_DIR / 'feature_scaler.pkl')
        
        with open(PROCESSED_DATA_DIR / 'feature_list.json', 'r') as f:
            self.feature_list = json.load(f)['features']
        
        print("✅ All resources loaded!")
    
    def extract_features(self, correct_word, incorrect_word):
        """Extract features"""
        features = {}
        
        features['edit_distance'] = Levenshtein.distance(correct_word, incorrect_word)
        features['word_length'] = len(correct_word)
        features['log_frequency'] = np.log1p(self.word_freq.get(correct_word.lower(), 0))
        
        vowels = sum(1 for c in correct_word.lower() if c in 'aeiou')
        features['num_vowels'] = vowels
        features['num_consonants'] = len(correct_word) - vowels
        features['vowel_ratio'] = vowels / len(correct_word) if len(correct_word) > 0 else 0
        features['consonant_ratio'] = features['num_consonants'] / len(correct_word) if len(correct_word) > 0 else 0
        
        error_pos = -1
        for i in range(min(len(correct_word), len(incorrect_word))):
            if i < len(correct_word) and i < len(incorrect_word):
                if correct_word[i] != incorrect_word[i]:
                    error_pos = i
                    break
        
        if error_pos == -1:
            error_pos = min(len(correct_word), len(incorrect_word))
        
        features['error_position'] = error_pos
        features['error_position_ratio'] = error_pos / len(correct_word) if len(correct_word) > 0 else 0
        features['error_on_vowel'] = 1 if error_pos < len(correct_word) and correct_word[error_pos] in 'aeiou' else 0
        features['error_at_beginning'] = 1 if error_pos < len(correct_word) * 0.3 else 0
        
        freq = self.word_freq.get(correct_word.lower(), 0)
        if freq == 0:
            features['freq_bin'] = 5
        elif freq < 50:
            features['freq_bin'] = 0
        elif freq < 200:
            features['freq_bin'] = 1
        elif freq < 1000:
            features['freq_bin'] = 2
        elif freq < 5000:
            features['freq_bin'] = 3
        else:
            features['freq_bin'] = 4
        
        first_char = correct_word[0] if correct_word else 'a'
        last_char = correct_word[-1] if correct_word else 'a'
        first_bigram = correct_word[:2] if len(correct_word) >= 2 else 'aa'
        last_bigram = correct_word[-2:] if len(correct_word) >= 2 else 'aa'
        first_trigram = correct_word[:3] if len(correct_word) >= 3 else 'aaa'
        last_trigram = correct_word[-3:] if len(correct_word) >= 3 else 'aaa'

        features['first_char_encoded'] = ord(first_char.lower()) - ord('a')
        features['last_char_encoded'] = ord(last_char.lower()) - ord('a')
        features['first_bigram_encoded'] = sum(ord(c) for c in first_bigram.lower()) % 676
        features['last_bigram_encoded'] = sum(ord(c) for c in last_bigram.lower()) % 676
        features['first_trigram_encoded'] = sum(ord(c) for c in first_trigram.lower()) % 17576
        features['last_trigram_encoded'] = sum(ord(c) for c in last_trigram.lower()) % 17576
        features['first_bigram_common'] = 1 if first_bigram.lower() in ['th', 'in', 'an', 'er', 'on', 're'] else 0
        features['last_bigram_common'] = 1 if last_bigram.lower() in ['ed', 'er', 'ly', 'ng', 'al', 'on'] else 0
        
        features['has_double_letters'] = 1 if any(correct_word[i] == correct_word[i+1] for i in range(len(correct_word)-1)) else 0
        features['num_double_letters'] = sum(1 for i in range(len(correct_word)-1) if correct_word[i] == correct_word[i+1])

        vowels_list = [c for c in correct_word.lower() if c in 'aeiou']
        features['has_repeated_vowels'] = 1 if any(vowels_list[i] == vowels_list[i+1] for i in range(len(vowels_list)-1)) else 0

        # Alternating pattern (CVCV)
        def is_alternating_pattern(word):
            if len(word) < 3:
                return 0
            pattern = ['V' if c in 'aeiou' else 'C' for c in word.lower()]
            for i in range(len(pattern)-1):
                if pattern[i] == pattern[i+1]:
                    return 0
            return 1
        features['is_alternating'] = is_alternating_pattern(correct_word)
        
        features['levenshtein_ratio'] = 1 - (features['edit_distance'] / max(len(correct_word), len(incorrect_word)))
        features['jaro_similarity'] = Levenshtein.jaro_winkler(correct_word, incorrect_word)
        
        syllables = 0
        prev_vowel = False
        for char in correct_word.lower():
            is_vowel = char in 'aeiou'
            if is_vowel and not prev_vowel:
                syllables += 1
            prev_vowel = is_vowel
        if correct_word.lower().endswith('e'):
            syllables -= 1
        syllables = max(1, syllables)
        
        features['syllable_count'] = syllables
        features['syllable_ratio'] = syllables / len(correct_word)
        
        max_cluster = 0
        current_cluster = 0
        for char in correct_word.lower():
            if char not in 'aeiou':
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0
        features['max_consonant_cluster'] = max_cluster

        # Max vowel cluster
        max_vowel_cluster = 0
        current_vowel_cluster = 0
        for char in correct_word.lower():
            if char in 'aeiou':
                current_vowel_cluster += 1
                max_vowel_cluster = max(max_vowel_cluster, current_vowel_cluster)
            else:
                current_vowel_cluster = 0
        features['max_vowel_cluster'] = max_vowel_cluster

        features['char_diversity'] = len(set(correct_word)) / len(correct_word) if len(correct_word) > 0 else 0
        features['unique_vowels'] = len(set(c for c in correct_word.lower() if c in 'aeiou'))
        features['unique_consonants'] = len(set(c for c in correct_word.lower() if c not in 'aeiou'))

        features['starts_with_vowel'] = 1 if correct_word and correct_word[0].lower() in 'aeiou' else 0
        features['ends_with_vowel'] = 1 if correct_word and correct_word[-1].lower() in 'aeiou' else 0

        if len(correct_word) > 0:
            mid = len(correct_word) // 2
            middle_char = correct_word[mid]
            features['middle_char_encoded'] = ord(middle_char.lower()) - ord('a')
            features['middle_is_vowel'] = 1 if middle_char.lower() in 'aeiou' else 0
        else:
            features['middle_char_encoded'] = 0
            features['middle_is_vowel'] = 0

        return features
    
    def find_corrections(self, word, max_distance=2, top_n=5):
        word_lower = word.lower()
        
        if len(word) == 1 and word_lower in ['a', 'i']:
            if word_lower in self.dictionary:
                return [(word, 0, self.word_freq.get(word_lower, 0), True, 100000)]
        
        if word_lower in self.dictionary:
            return [(word, 0, self.word_freq.get(word_lower, 0), True, 100000)]
        
        candidates = []
        
        for dict_word in self.dictionary:
            dist = Levenshtein.distance(word_lower, dict_word)
            
            if dist <= max_distance:
                freq = self.word_freq.get(dict_word, 0)
                
                score = 0
                score += (max_distance - dist + 1) * 10000
                score += min(freq, 50000) / 10
                
                if len(dict_word) == len(word_lower):
                    score += 5000
                elif abs(len(dict_word) - len(word_lower)) == 1:
                    score += 2000
                
                if dict_word[0] == word_lower[0]:
                    score += 3000
                
                if dict_word[-1] == word_lower[-1]:
                    score += 1500
                
                jaro = Levenshtein.jaro_winkler(word_lower, dict_word)
                score += jaro * 2000
                
                candidates.append((dict_word, dist, freq, False, score))
        
        if not candidates:
            return [(word, -1, 0, False, 0)]
        
        candidates.sort(key=lambda x: -x[4])
        
        return candidates[:top_n]
    
    def predict_error_type(self, correct_word, incorrect_word, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        features_dict = self.extract_features(correct_word, incorrect_word)
        features_df = pd.DataFrame([features_dict])[self.feature_list]
        features_normalized = self.scaler.transform(features_df)
        
        model = self.models[model_name]
        prediction = model.predict(features_normalized)[0]
        proba = model.predict_proba(features_normalized)[0]
        
        error_type = self.label_names[prediction]
        confidence = float(proba[prediction])
        
        prob_dist = {self.label_names[i]: float(proba[i]) for i in range(len(proba))}
        
        return error_type, confidence, prob_dist

checker = SpellCheckerAPI()

@app.get("/")
async def root():
    return {
        "message": "AI Spell Checker API - Anti-Overfitting",
        "version": "2.0.0",
        "accuracy": "84%",
        "overfitting": "<5%",
        "endpoints": {
            "check_word": "/api/check-word",
            "check_sentence": "/api/check-sentence",
            "models": "/api/models",
            "stats": "/api/stats"
        }
    }

@app.get("/api/models")
async def get_models():
    return {
        "models": [
            {"name": "Ensemble", "accuracy": "84.67%", "overfitting": "3.5%", "description": "Best: Weighted voting"},
            {"name": "XGBoost", "accuracy": "84.60%", "overfitting": "4.8%", "description": "Fast & balanced"},
            {"name": "LightGBM", "accuracy": "84.53%", "overfitting": "3.4%", "description": "Fastest"},
            {"name": "RandomForest", "accuracy": "80.00%", "overfitting": "0.6%", "description": "Most stable"}
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """Get system stats"""
    return {
        "dictionary_size": len(checker.dictionary),
        "models_loaded": len(checker.models),
        "features": len(checker.feature_list),
        "error_types": list(checker.label_names.values())
    }

@app.post("/api/check-word", response_model=CheckWordResponse)
async def check_word(request: CheckWordRequest):
    """Check a single word"""
    try:
        word = request.word.strip()
        
        if not word:
            raise HTTPException(status_code=400, detail="Word cannot be empty")
        
        corrections = checker.find_corrections(word, max_distance=2, top_n=5)
        
        is_correct = corrections[0][3]
        
        response = {
            "is_correct": is_correct,
            "word": word,
            "corrections": []
        }
        
        if not is_correct and corrections[0][1] != -1:
            # Get error type prediction
            best_correction = corrections[0][0]
            error_type, confidence, prob_dist = checker.predict_error_type(
                best_correction, word, request.model_name
            )
            
            response["error_type"] = error_type
            response["confidence"] = confidence
            response["probability_distribution"] = prob_dist
            
            # Add corrections
            for corr_word, dist, freq, _, score in corrections:
                response["corrections"].append({
                    "word": corr_word,
                    "distance": dist,
                    "frequency": freq,
                    "score": score
                })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-sentence", response_model=CheckSentenceResponse)
async def check_sentence(request: CheckSentenceRequest):
    """Check a sentence"""
    try:
        sentence = request.sentence.strip()
        
        if not sentence:
            raise HTTPException(status_code=400, detail="Sentence cannot be empty")
        
        words = re.findall(r'\b\w+\b|[^\w\s]', sentence)
        
        errors = []
        corrected_words = []
        common_single_chars = {'a', 'i', 'A', 'I'}
        
        for word in words:
            if not word.isalpha():
                corrected_words.append(word)
                continue
            
            if word in common_single_chars:
                corrected_words.append(word)
                continue
            
            word_lower = word.lower()
            
            if word_lower in checker.dictionary:
                corrected_words.append(word)
                continue
            
            corrections = checker.find_corrections(word, max_distance=2, top_n=1)
            
            if corrections[0][3]:
                corrected_words.append(word)
            else:
                if corrections[0][1] != -1:
                    best_correction = corrections[0][0]
                    
                    if word[0].isupper():
                        best_correction = best_correction.capitalize()
                    
                    error_type, confidence, _ = checker.predict_error_type(
                        corrections[0][0], word_lower, request.model_name
                    )
                    
                    # Find position in original sentence
                    position = sentence.lower().find(word_lower)
                    
                    errors.append({
                        "original": word,
                        "correction": best_correction,
                        "error_type": error_type,
                        "confidence": confidence,
                        "distance": corrections[0][1],
                        "position": position
                    })
                    
                    corrected_words.append(best_correction)
                else:
                    corrected_words.append(word)
        
        corrected_sentence = ' '.join(corrected_words)
        corrected_sentence = re.sub(r'\s+([.,!?;:])', r'\1', corrected_sentence)
        
        return {
            "original_sentence": sentence,
            "corrected_sentence": corrected_sentence,
            "has_errors": len(errors) > 0,
            "num_errors": len(errors),
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)