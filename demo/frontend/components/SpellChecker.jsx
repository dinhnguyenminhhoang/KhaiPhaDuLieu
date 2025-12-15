'use client'

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Check, X, AlertCircle, Zap, Brain, Settings,
    ArrowRight, TrendingUp, BookOpen, Sparkles
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function SpellChecker() {
    // States
    const [inputText, setInputText] = useState('');
    const [outputText, setOutputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('LightGBM');
    const [models, setModels] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(false);
    const [checkMode, setCheckMode] = useState('sentence');
    const [result, setResult] = useState(null);
    const [errors, setErrors] = useState([]);

    // Loading states
    const [modelsLoading, setModelsLoading] = useState(true);
    const [statsLoading, setStatsLoading] = useState(true);
    const [loadError, setLoadError] = useState(null);

    // Load data on mount
    useEffect(() => {
        loadModels();
        loadStats();
    }, []);

    const loadModels = async () => {
        try {
            setModelsLoading(true);
            const response = await axios.get(`${API_BASE}/api/models`);
            setModels(response.data.models);
            setLoadError(null);
        } catch (error) {
            console.error('Error loading models:', error);
            setLoadError('Failed to load models. Please check if backend is running.');
        } finally {
            setModelsLoading(false);
        }
    };

    const loadStats = async () => {
        try {
            setStatsLoading(true);
            const response = await axios.get(`${API_BASE}/api/stats`);
            setStats(response.data);
        } catch (error) {
            console.error('Error loading stats:', error);
            // Don't override error if models already failed
            if (!loadError) {
                setLoadError('Failed to load system stats.');
            }
        } finally {
            setStatsLoading(false);
        }
    };

    const handleCheck = async () => {
        if (!inputText.trim()) {
            alert('Please enter text to check');
            return;
        }

        setLoading(true);
        setResult(null);
        setErrors([]);

        try {
            if (checkMode === 'word') {
                const response = await axios.post(`${API_BASE}/api/check-word`, {
                    word: inputText.trim(),
                    model_name: selectedModel
                });

                setResult(response.data);

                if (response.data.is_correct) {
                    setOutputText(`✅ "${inputText.trim()}" is correct!`);
                } else {
                    // Chỉ lấy kết quả chính xác nhất (correction đầu tiên)
                    const bestCorrection = response.data.corrections[0]?.word || inputText;
                    setOutputText(bestCorrection);
                }
            } else {
                const response = await axios.post(`${API_BASE}/api/check-sentence`, {
                    sentence: inputText.trim(),
                    model_name: selectedModel
                });

                setResult(response.data);
                setOutputText(response.data.corrected_sentence);
                setErrors(response.data.errors || []);
            }
        } catch (error) {
            console.error('Error checking:', error);
            alert('Error: ' + (error.response?.data?.detail || error.message));
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setInputText('');
        setOutputText('');
        setResult(null);
        setErrors([]);
    };

    const handleExample = () => {
        if (checkMode === 'word') {
            // Examples for single word mode
            const wordExamples = [
                'recieve',
                'neccessary',
                'seperate',
                'definate',
                'occurance',
                'truely',
                'beleive',
                'acheive',
                'accomodation',
                'convinient',
                'grammer',
                'existance',
                'wierd',
                'publically',
                'occured'
            ];
            setInputText(wordExamples[Math.floor(Math.random() * wordExamples.length)]);
        } else {
            // Examples for sentence mode
            const sentenceExamples = [
                'I will recieve the neccessary package tomorrow',
                'This is seperate from the definate answer',
                'The occurance was truely remarkable',
                'He beleives in acheiving his goals',
                'The accomodation was quite convinient'
            ];
            setInputText(sentenceExamples[Math.floor(Math.random() * sentenceExamples.length)]);
        }
    };

    const handleCopyOutput = () => {
        if (outputText) {
            navigator.clipboard.writeText(outputText);
            alert('Copied to clipboard!');
        }
    };

    return (
        <div className="app">
            {/* Error Notification */}
            {loadError && (
                <div className="error-notification">
                    <AlertCircle size={20} />
                    <span>{loadError}</span>
                    <button onClick={() => { loadModels(); loadStats(); }} className="retry-btn">
                        🔄 Retry
                    </button>
                </div>
            )}

            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="header-left">
                        <Brain className="logo-icon" size={32} />
                        <div>
                            <h1>AI Spell Checker</h1>
                        </div>
                    </div>

                    <div className="header-right">
                        <div className="model-selector">
                            <Settings size={18} />
                            {modelsLoading ? (
                                <div className="skeleton-select"></div>
                            ) : (
                                <select
                                    value={selectedModel}
                                    onChange={(e) => setSelectedModel(e.target.value)}
                                    className="model-select"
                                    disabled={models.length === 0}
                                >
                                    {models.map(model => (
                                        <option key={model.name} value={model.name}>
                                            {model.name} ({model.accuracy})
                                        </option>
                                    ))}
                                </select>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Stats Bar */}
            {statsLoading ? (
                <div className="stats-bar">
                    <div className="stat-item">
                        <div className="skeleton-stat"></div>
                    </div>
                    <div className="stat-item">
                        <div className="skeleton-stat"></div>
                    </div>
                    <div className="stat-item">
                        <div className="skeleton-stat"></div>
                    </div>
                    <div className="stat-item">
                        <div className="skeleton-stat"></div>
                    </div>
                </div>
            ) : stats && (
                <div className="stats-bar">
                    <div className="stat-item">
                        <BookOpen size={16} />
                        <span>{stats.dictionary_size.toLocaleString()} words</span>
                    </div>
                    <div className="stat-item">
                        <Brain size={16} />
                        <span>{stats.models_loaded} models</span>
                    </div>
                    <div className="stat-item">
                        <TrendingUp size={16} />
                        <span>{stats.features} features</span>
                    </div>
                    <div className="stat-item">
                        <Zap size={16} />
                        <span>100K samples trained</span>
                    </div>
                </div>
            )}

            {/* Main Content */}
            <div className="container">
                {/* Mode Toggle */}
                <div className="mode-toggle">
                    <button
                        className={`mode-btn ${checkMode === 'word' ? 'active' : ''}`}
                        onClick={() => setCheckMode('word')}
                    >
                        🔤 Single Word
                    </button>
                    <button
                        className={`mode-btn ${checkMode === 'sentence' ? 'active' : ''}`}
                        onClick={() => setCheckMode('sentence')}
                    >
                        📝 Full Sentence
                    </button>
                </div>

                {/* Input/Output Panels */}
                <div className="panels">
                    {/* Input Panel */}
                    <div className="panel input-panel">
                        <div className="panel-header">
                            <h3>📥 Input</h3>
                            <button onClick={handleExample} className="btn-example">
                                <Sparkles size={14} /> Try Example
                            </button>
                        </div>
                        <textarea
                            className="textarea"
                            placeholder={checkMode === 'word'
                                ? "Enter a word to check...\ne.g., recieve, neccessary, seperate"
                                : "Enter a sentence to check...\ne.g., I will recieve the neccessary package tomorrow"}
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            rows={10}
                        />

                        <div className="panel-footer">
                            <button
                                onClick={handleCheck}
                                disabled={loading || !inputText.trim()}
                                className="btn-check"
                            >
                                {loading ? (
                                    <>
                                        <Zap size={18} className="spin" />
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <Check size={18} />
                                        Check Spelling
                                    </>
                                )}
                            </button>
                            <button onClick={handleClear} className="btn-clear">
                                <X size={18} /> Clear
                            </button>
                        </div>
                    </div>

                    {/* Arrow */}
                    <div className="arrow-container">
                        <div className="arrow-circle">
                            <ArrowRight size={28} className="arrow-icon" />
                        </div>
                    </div>

                    {/* Output Panel */}
                    <div className="panel output-panel">
                        <div className="panel-header">
                            <h3>📤 Output</h3>
                            <div className="panel-header-right">
                                {result && (
                                    <span className={`status-badge ${checkMode === 'word'
                                            ? (result.is_correct ? 'correct' : 'incorrect')
                                            : (result.has_errors ? 'incorrect' : 'correct')
                                        }`}>
                                        {checkMode === 'word'
                                            ? (result.is_correct ? '✓ Correct' : '✗ Incorrect')
                                            : (result.has_errors ? `${result.num_errors} Error(s)` : '✓ No Errors')}
                                    </span>
                                )}
                                {outputText && (
                                    <button onClick={handleCopyOutput} className="btn-copy">
                                        Copy
                                    </button>
                                )}
                            </div>
                        </div>

                        <textarea
                            className="textarea output-textarea"
                            placeholder="Corrected text will appear here..."
                            value={outputText}
                            readOnly
                            rows={10}
                        />

                        {/* Suggestions for Word */}
                        {checkMode === 'word' && result && !result.is_correct && result.corrections && (
                            <div className="suggestions">
                                <h4>💡 Suggestions:</h4>
                                <div className="suggestion-list">
                                    {result.corrections.slice(0, 5).map((correction, idx) => (
                                        <div
                                            key={idx}
                                            className={`suggestion-item ${idx === 0 ? 'best' : ''}`}
                                            onClick={() => setOutputText(correction.word)}
                                        >
                                            {idx === 0 && <span className="best-badge">BEST</span>}
                                            <span className="suggestion-word">{correction.word}</span>
                                            <span className="suggestion-meta">
                                                Dist: {correction.distance} |
                                                Freq: {correction.frequency.toLocaleString()}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Error Details for Sentence */}
                {checkMode === 'sentence' && errors.length > 0 && (
                    <div className="error-details">
                        <h3>🔍 Error Analysis ({errors.length} error{errors.length > 1 ? 's' : ''} found)</h3>
                        <div className="error-grid">
                            {errors.map((error, idx) => (
                                <div key={idx} className="error-card">
                                    <div className="error-card-header">
                                        <span className="error-number">#{idx + 1}</span>
                                        <span className={`error-type-badge ${error.error_type}`}>
                                            {error.error_type.toUpperCase()}
                                        </span>
                                    </div>

                                    <div className="error-words">
                                        <div className="error-word-item">
                                            <span className="label">Original:</span>
                                            <span className="original">{error.original}</span>
                                        </div>
                                        <ArrowRight size={20} className="arrow-small" />
                                        <div className="error-word-item">
                                            <span className="label">Correction:</span>
                                            <span className="correction">{error.correction}</span>
                                        </div>
                                    </div>

                                    <div className="error-stats">
                                        <div className="stat">
                                            <span className="stat-label">Confidence</span>
                                            <div className="confidence-bar-container">
                                                <div
                                                    className="confidence-bar"
                                                    style={{ width: `${error.confidence * 100}%` }}
                                                />
                                            </div>
                                            <span className="stat-value">{(error.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="stat">
                                            <span className="stat-label">Edit Distance</span>
                                            <span className="stat-value">{error.distance}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Probability Distribution for Word */}
                {checkMode === 'word' && result && !result.is_correct && result.probability_distribution && (
                    <div className="probability-panel">
                        <h3>🤖 AI Analysis - Error Type Prediction</h3>
                        <p className="probability-subtitle">
                            Model: <strong>{selectedModel}</strong>
                        </p>
                        <div className="probability-bars">
                            {Object.entries(result.probability_distribution)
                                .sort((a, b) => b[1] - a[1])
                                .map(([type, prob]) => (
                                    <div key={type} className="prob-item">
                                        <div className="prob-header">
                                            <span className={`prob-type ${type}`}>{type}</span>
                                            <span className="prob-value">{(prob * 100).toFixed(2)}%</span>
                                        </div>
                                        <div className="prob-bar-container">
                                            <div
                                                className={`prob-bar ${type}`}
                                                style={{ width: `${prob * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}