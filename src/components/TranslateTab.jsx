import React, { useState, useRef, useCallback } from 'react';
import { CameraView } from './CameraView';
import { normalizeLandmarks } from '../ml/preprocessor';
import { SpeakerIcon, TrashIcon, UndoIcon, BrainIcon } from './Icons';

export function TranslateTab({
  knnModel,
  tfModel,
  modelStatus,
  activeModelType,
  setActiveModelType,
  onGoToTrain,
}) {
  // Speech synthesis toggle
  const [speakAuto, setSpeakAuto] = useState(false);

  // Real-time inference state
  const [instantPrediction, setInstantPrediction] = useState(null); // { label, confidence, allConfidences }
  const [stablePrediction, setStablePrediction] = useState(null);   // { label, confidence }
  const [sentenceWords, setSentenceWords] = useState([]);
  const [speakingWord, setSpeakingWord] = useState(null);

  // Smoothing buffers
  const historyBufferRef = useRef([]); // Last 15 frames { label, confidence }
  const lastAddedSignRef = useRef(null);
  const handPresentRef = useRef(false);

  // Web Speech API helper
  const speakText = useCallback((text) => {
    if (!('speechSynthesis' in window) || !text) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.92;
      utterance.pitch = 1.05;
      utterance.onstart = () => setSpeakingWord(text);
      utterance.onend = () => setSpeakingWord(null);
      utterance.onerror = () => setSpeakingWord(null);
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis failed:', e);
    }
  }, []);

  // Process incoming video frame landmarks
  const handleFrameLandmarks = useCallback((landmarks) => {
    if (!landmarks) {
      // Hand has left the frame: reset latch so next sign can be captured
      if (handPresentRef.current) {
        handPresentRef.current = false;
        lastAddedSignRef.current = null;
        historyBufferRef.current = [];
        setInstantPrediction(null);
        setStablePrediction(null);
      }
      return;
    }

    handPresentRef.current = true;

    // Check if any model is trained
    const isMlpReady = tfModel && tfModel.isTrained;
    const isKnnReady = knnModel && knnModel.isTrained;

    if (!isMlpReady && !isKnnReady) {
      setInstantPrediction(null);
      return;
    }

    try {
      // Step 1: Preprocess raw landmarks to normalized 130-feature vector
      const vector = normalizeLandmarks(landmarks);
      if (!vector) {
        setInstantPrediction(null);
        return;
      }

      // Step 2: Infer using active model
      let result = null;
      if (activeModelType === 'mlp' && isMlpReady) {
        result = tfModel.predict(vector);
      } else if (isKnnReady) {
        result = knnModel.predict(vector);
      } else if (isMlpReady) {
        result = tfModel.predict(vector);
      }

      if (!result || result.classIndex === -1) {
        setInstantPrediction(null);
        return;
      }

      setInstantPrediction(result);

      // Step 3: Rolling window majority vote over last 15 frames
      const buffer = historyBufferRef.current;
      buffer.push(result);
      if (buffer.length > 15) {
        buffer.shift();
      }

      // Filter frames with confidence > 0.80
      const confidentVotes = {};
      let validCount = 0;

      for (const item of buffer) {
        if (item.confidence >= 0.80) {
          confidentVotes[item.label] = (confidentVotes[item.label] || 0) + 1;
          validCount++;
        }
      }

      // Check if any label has >= 10 votes
      let winningLabel = null;
      let maxVotes = 0;

      for (const [lbl, count] of Object.entries(confidentVotes)) {
        if (count >= 10 && count > maxVotes) {
          maxVotes = count;
          winningLabel = lbl;
        }
      }

      if (winningLabel) {
        const stableObj = {
          label: winningLabel,
          confidence: result.confidence,
        };
        setStablePrediction(stableObj);

        // Auto-build sentence (prevent repeat until hand leaves frame or sign changes)
        if (lastAddedSignRef.current !== winningLabel) {
          lastAddedSignRef.current = winningLabel;
          setSentenceWords(prev => [...prev, winningLabel]);

          if (speakAuto) {
            speakText(winningLabel);
          }
        }
      } else {
        // Not stable yet
        setStablePrediction(null);
      }
    } catch (err) {
      console.warn('Inference frame error:', err);
    }
  }, [activeModelType, tfModel, knnModel, speakAuto, speakText]);

  // Sentence controls
  const handleSpeakSentence = () => {
    if (sentenceWords.length === 0) return;
    speakText(sentenceWords.join(' '));
  };

  const handleClearSentence = () => {
    setSentenceWords([]);
    lastAddedSignRef.current = null;
  };

  const handleUndoSentence = () => {
    setSentenceWords(prev => prev.slice(0, -1));
    lastAddedSignRef.current = null;
  };

  // Determine current status message
  let feedbackMessage = 'Hold your hand steadily in front of the lens';
  let isConfident = false;

  if (!modelStatus.trained) {
    feedbackMessage = 'No trained model available yet. Visit Teach or Train tab!';
  } else if (!instantPrediction) {
    feedbackMessage = 'Waiting for hand gesture...';
  } else if (stablePrediction) {
    feedbackMessage = `Got it: ${stablePrediction.label}`;
    isConfident = true;
  } else if (instantPrediction.confidence >= 0.80) {
    feedbackMessage = `Holding "${instantPrediction.label}" — stabilizing...`;
  } else {
    feedbackMessage = 'Hmm, not sure yet — try holding gesture closer or more clearly';
  }

  const confidencePercent = instantPrediction
    ? Math.round(instantPrediction.confidence * 100)
    : 0;

  return (
    <div className="translate-layout">
      {/* Left Column: Polaroid Camera Feed */}
      <section className="camera-column" aria-label="Camera Feed">
        <CameraView
          onFrameLandmarks={handleFrameLandmarks}
          polaroidCaption="MediaPipe 21 Hand Keypoints Overlay"
        />

        {/* Model Switcher Pill */}
        <div className="model-selector-bar">
          <span className="selector-label">Inference Model:</span>
          <div className="selector-pills">
            <button
              type="button"
              className={`pill-btn ${activeModelType === 'mlp' ? 'pill-btn-active' : ''}`}
              onClick={() => setActiveModelType('mlp')}
              disabled={!tfModel || !tfModel.isTrained}
              title={tfModel && tfModel.isTrained ? 'Use Deep Neural Network' : 'MLP not trained yet'}
            >
              Neural Network (MLP)
            </button>
            <button
              type="button"
              className={`pill-btn ${activeModelType === 'knn' ? 'pill-btn-active' : ''}`}
              onClick={() => setActiveModelType('knn')}
              disabled={!knnModel || !knnModel.isTrained}
              title={knnModel && knnModel.isTrained ? 'Use k-NN baseline' : 'k-NN not trained yet'}
            >
              k-NN Baseline (k=5)
            </button>
          </div>
        </div>
      </section>

      {/* Right Column: Real-time Recognition & Sentence Builder */}
      <section className="output-column" aria-label="Recognition Output">
        {/* Real-time Sign Card */}
        <div className="sianor-card recognition-card">
          <div className="card-header-row">
            <span className="card-kicker">Live Prediction</span>
            <span className={`confidence-tag ${isConfident ? 'tag-confident' : 'tag-neutral'}`}>
              {instantPrediction ? `${confidencePercent}% Match` : 'Idle'}
            </span>
          </div>

          <div className="prediction-display">
            {stablePrediction ? (
              <div className="word-hero">
                <span className="word-text word-stable">{stablePrediction.label}</span>
                <span className="word-indicator-stable">Stable (10+ frames)</span>
              </div>
            ) : instantPrediction && instantPrediction.confidence >= 0.80 ? (
              <div className="word-hero">
                <span className="word-text word-tentative">{instantPrediction.label}</span>
                <span className="word-indicator-tentative">Stabilizing...</span>
              </div>
            ) : (
              <div className="word-hero">
                <span className="word-placeholder">
                  {modelStatus.trained ? '—' : 'Untrained'}
                </span>
                <span className="word-indicator-none">
                  {modelStatus.trained ? 'Awaiting sign' : 'Need training data'}
                </span>
              </div>
            )}
          </div>

          {/* Confidence Meter Bar */}
          <div className="meter-wrapper">
            <div className="meter-labels">
              <span>Confidence Threshold (&gt;80%)</span>
              <span className="meter-val">{confidencePercent}%</span>
            </div>
            <div className="meter-track">
              <div
                className={`meter-fill ${confidencePercent >= 80 ? 'meter-fill-high' : 'meter-fill-low'}`}
                style={{ width: `${confidencePercent}%` }}
              />
              {/* 80% threshold line indicator */}
              <div className="meter-threshold-marker" style={{ left: '80%' }} title="80% Acceptance Threshold" />
            </div>
          </div>

          {/* Human Friendly Status Microcopy */}
          <div className="feedback-strip">
            <p className="feedback-text">{feedbackMessage}</p>
          </div>

          {/* Candidate Probabilities Mini-Breakdown */}
          {instantPrediction && instantPrediction.allConfidences && (
            <div className="candidate-drawer">
              <span className="candidate-heading">Top candidates:</span>
              <div className="candidate-bars">
                {Object.entries(instantPrediction.allConfidences)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 3)
                  .map(([name, conf]) => (
                    <div key={name} className="candidate-item">
                      <span className="candidate-name">{name}</span>
                      <div className="candidate-mini-bar">
                        <div
                          className="candidate-mini-fill"
                          style={{ width: `${Math.round(conf * 100)}%` }}
                        />
                      </div>
                      <span className="candidate-num">{Math.round(conf * 100)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {!modelStatus.trained && (
            <div className="untrained-prompt">
              <p>No gesture model trained yet. Load the demo dataset or record your own signs!</p>
              <button
                type="button"
                className="sianor-btn sianor-btn-primary"
                onClick={onGoToTrain}
              >
                <BrainIcon size={16} />
                <span>Go to Train Tab</span>
              </button>
            </div>
          )}
        </div>

        {/* Sentence Builder Card */}
        <div className="sianor-card sentence-card">
          <div className="card-header-row">
            <div>
              <span className="card-kicker">Sentence Assembly</span>
              <h3 className="card-title-sm">Translated Text</h3>
            </div>
            <label className="toggle-label" title="Automatically speak each recognized sign aloud">
              <input
                type="checkbox"
                checked={speakAuto}
                onChange={e => setSpeakAuto(e.target.checked)}
                className="sianor-toggle"
              />
              <span className="toggle-text">Speak automatically</span>
            </label>
          </div>

          {/* Sentence Display Paper Area */}
          <div className="sentence-paper" tabIndex={0} aria-label="Constructed sentence">
            {sentenceWords.length > 0 ? (
              <div className="word-chips">
                {sentenceWords.map((word, idx) => (
                  <span
                    key={`${word}-${idx}`}
                    className={`word-chip ${speakingWord === word ? 'word-chip-speaking' : ''}`}
                  >
                    {word}
                  </span>
                ))}
              </div>
            ) : (
              <p className="sentence-empty">
                Stable gestures will automatically appear here to form words and sentences.
              </p>
            )}
          </div>

          {/* Action Button Row */}
          <div className="sentence-actions">
            <button
              type="button"
              className="sianor-btn sianor-btn-primary"
              onClick={handleSpeakSentence}
              disabled={sentenceWords.length === 0}
              title="Speak assembled sentence aloud"
            >
              <SpeakerIcon size={18} />
              <span>Speak</span>
            </button>

            <button
              type="button"
              className="sianor-btn sianor-btn-secondary"
              onClick={handleUndoSentence}
              disabled={sentenceWords.length === 0}
              title="Remove last word"
            >
              <UndoIcon size={16} />
              <span>Undo</span>
            </button>

            <button
              type="button"
              className="sianor-btn sianor-btn-ghost"
              onClick={handleClearSentence}
              disabled={sentenceWords.length === 0}
              title="Clear all words"
            >
              <TrashIcon size={16} />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
