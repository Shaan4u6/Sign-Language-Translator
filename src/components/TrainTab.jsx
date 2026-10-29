import React, { useState } from 'react';
import { stratifiedSplit, computeEvaluationMetrics } from '../ml/metrics';
import { augmentDataset } from '../ml/preprocessor';
import { KNNClassifier } from '../ml/knn';
import { TFHandModel } from '../ml/tfModel';
import { TrainingCharts } from './TrainingCharts';
import { ConfusionMatrix } from './ConfusionMatrix';
import { BrainIcon, CheckIcon } from './Icons';
import confetti from 'canvas-confetti';

export function TrainTab({
  dataset,
  setKnnModel,
  setTfModel,
  setModelStatus,
  onGoToTranslate,
}) {
  // Hyperparameters & Toggles
  const [useAugmentation, setUseAugmentation] = useState(true);
  const [epochs, setEpochs] = useState(50);
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [epochHistory, setEpochHistory] = useState([]);

  // Post-training Evaluation Results
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [savedToIndexedDb, setSavedToIndexedDb] = useState(false);

  // Train workflow
  const handleTrainModels = async () => {
    if (dataset.length < 10) {
      alert('You need at least 10 samples to train. Please record gestures or click "Load Demo Dataset" in Teach tab.');
      return;
    }

    setIsTraining(true);
    setEpochHistory([]);
    setCurrentEpoch(0);
    setEvaluationResults(null);
    setSavedToIndexedDb(false);

    try {
      // Step 1: Stratified 80/20 train/test split
      const split = stratifiedSplit(dataset, 0.8);
      const { X_train: rawXTrain, y_train: rawYTrain, X_test, y_test, labels: splitLabels } = split;

      // Step 2: Data Augmentation (if enabled)
      let finalXTrain = rawXTrain;
      let finalYTrain = rawYTrain;

      if (useAugmentation) {
        const trainItems = rawXTrain.map((feat, i) => ({
          features: feat,
          label: splitLabels[rawYTrain[i]],
        }));
        const augmentedItems = augmentDataset(trainItems, 2); // 2 synthetic copies per sample

        const labelToIdx = new Map(splitLabels.map((l, i) => [l, i]));
        finalXTrain = augmentedItems.map(item => item.features);
        finalYTrain = augmentedItems.map(item => labelToIdx.get(item.label));
      }

      // Step 3: Train k-NN Baseline (k=5)
      const knn = new KNNClassifier(5);
      knn.fit(finalXTrain, finalYTrain, splitLabels);
      const knnTestPreds = knn.predictAll(X_test);
      const knnMetrics = computeEvaluationMetrics(y_test, knnTestPreds, splitLabels);

      // Step 4: Train TensorFlow.js MLP
      const mlp = new TFHandModel();
      await mlp.fit(
        finalXTrain,
        finalYTrain,
        X_test,
        y_test,
        splitLabels,
        {
          epochs,
          batchSize: 16,
          onEpochEnd: (log) => {
            setCurrentEpoch(log.epoch);
            setEpochHistory(prev => [...prev, log]);
          },
        }
      );

      // Step 5: Evaluate MLP on holdout test set
      const mlpTestPreds = mlp.predictAll(X_test);
      const mlpMetrics = computeEvaluationMetrics(y_test, mlpTestPreds, splitLabels);

      // Save best model (MLP) to IndexedDB
      try {
        await mlp.saveToIndexedDB('sianor-hand-mlp');
        setSavedToIndexedDb(true);
      } catch (saveErr) {
        console.warn('Could not save to IndexedDB:', saveErr);
      }

      // Update parent model states
      setKnnModel(knn);
      setTfModel(mlp);
      setModelStatus({
        trained: true,
        modelType: 'mlp',
        labels: splitLabels,
        accuracy: mlpMetrics.accuracy,
      });

      setEvaluationResults({
        knnMetrics,
        mlpMetrics,
        splitLabels,
        trainCount: finalXTrain.length,
        testCount: X_test.length,
        originalTrainCount: rawXTrain.length,
      });

      confetti({
        particleCount: 80,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#C85A32', '#E09F3E', '#FAF7F2'],
      });
    } catch (err) {
      console.error('Training error:', err);
      alert(`Training failed: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="train-page-container">
      {/* Top Banner: Mathematical Preprocessing Pipeline */}
      <section className="sianor-card preprocessing-card" aria-label="Feature Engineering Pipeline">
        <div className="card-header-row">
          <div>
            <span className="card-kicker">Feature Engineering</span>
            <h3 className="card-title-sm">Invariant Landmark Preprocessing</h3>
          </div>
          <span className="pipeline-badge">21 Keypoints → 63-Feature Vector</span>
        </div>

        <p className="card-instruction">
          Raw pixel coordinates vary wildly with camera distance, person height, and hand position.
          Before feeding data to our classifiers, every hand frame undergoes four normalization steps:
        </p>

        <div className="pipeline-steps-grid">
          <div className="step-card">
            <span className="step-num">a</span>
            <h4 className="step-title">Wrist Translation</h4>
            <p className="step-desc">
              All 21 landmarks are translated so Landmark 0 (wrist) sits at the coordinate origin <code>(0, 0, 0)</code>.
              Makes features translation-invariant.
            </p>
          </div>

          <div className="step-card">
            <span className="step-num">b</span>
            <h4 className="step-title">Max-Distance Scaling</h4>
            <p className="step-desc">
              Coordinates are divided by the maximum Euclidean distance from the wrist to any landmark.
              Ensures invariant size whether close or far.
            </p>
          </div>

          <div className="step-card">
            <span className="step-num">c</span>
            <h4 className="step-title">Handedness Flip</h4>
            <p className="step-desc">
              Left-hand samples have their X-coordinates reflected (<code>x = -x</code>) to mirror right-hand geometry.
              Both hands work seamlessly!
            </p>
          </div>

          <div className="step-card">
            <span className="step-num">d</span>
            <h4 className="step-title">Vector Flattening</h4>
            <p className="step-desc">
              The normalized 3D coordinates are flattened into an input array of length 63:
              <code>[x0, y0, z0, ..., x20, y20, z20]</code>.
            </p>
          </div>
        </div>
      </section>

      {/* Training Configuration & Launch Panel */}
      <section className="sianor-card train-config-card" aria-label="Training Setup">
        <div className="card-header-row">
          <div>
            <span className="card-kicker">Model Training</span>
            <h3 className="card-title-sm">Training Configuration</h3>
          </div>
          <div className="dataset-status-pill">
            <span>{dataset.length} total samples</span>
          </div>
        </div>

        <div className="config-grid">
          <div className="config-item">
            <span className="config-label">Dataset Split</span>
            <span className="config-val">80% Train / 20% Test (Stratified)</span>
            <span className="config-sub">Balanced class distribution across splits</span>
          </div>

          <div className="config-item">
            <span className="config-label">Data Augmentation</span>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={useAugmentation}
                onChange={e => setUseAugmentation(e.target.checked)}
                className="sianor-toggle"
                disabled={isTraining}
              />
              <span className="toggle-text">Rotation (±12°), Scale (±8%), Noise</span>
            </label>
            <span className="config-sub">Multiplies training data 3× with realistic jitter</span>
          </div>

          <div className="config-item">
            <span className="config-label">Neural Network Architecture</span>
            <span className="config-val">Dense 128 (ReLU) → Dropout 0.3 → Dense 64 → Softmax</span>
            <span className="config-sub">Adam Optimizer (lr=0.001) • Categorical Crossentropy</span>
          </div>

          <div className="config-item">
            <span className="config-label">Training Epochs</span>
            <div className="epoch-input-group">
              <input
                type="number"
                min={10}
                max={100}
                value={epochs}
                onChange={e => setEpochs(Math.max(10, Math.min(100, parseInt(e.target.value) || 50)))}
                className="sianor-input epoch-input"
                disabled={isTraining}
              />
              <span className="config-sub">Standard: 50 epochs</span>
            </div>
          </div>
        </div>

        <div className="train-launch-bar">
          <button
            type="button"
            className="sianor-btn sianor-btn-primary btn-lg"
            onClick={handleTrainModels}
            disabled={isTraining || dataset.length < 10}
          >
            <BrainIcon size={20} />
            <span>
              {isTraining
                ? `Training Epoch ${currentEpoch} of ${epochs}...`
                : 'Train k-NN & Neural Network Models'}
            </span>
          </button>

          {dataset.length < 10 && (
            <p className="train-warn-text">
              Add at least 10 gesture samples or load the demo dataset in the Teach tab before training.
            </p>
          )}
        </div>
      </section>

      {/* Live Training Curves (During or After Training) */}
      {(isTraining || epochHistory.length > 0) && (
        <section className="sianor-card training-curves-section" aria-label="Live Training Charts">
          <div className="card-header-row">
            <div>
              <span className="card-kicker">Real-Time Loss & Accuracy</span>
              <h3 className="card-title-sm">TF.js Live Training Progress</h3>
            </div>
            {isTraining && (
              <span className="training-active-tag">
                <span className="rec-pulse-dot" /> Epoch {currentEpoch}/{epochs}
              </span>
            )}
          </div>

          <TrainingCharts epochHistory={epochHistory} maxEpochs={epochs} />
        </section>
      )}

      {/* Post-Training Comprehensive Evaluation Dashboard */}
      {evaluationResults && (
        <section className="sianor-card results-dashboard-card" aria-label="Evaluation Dashboard">
          <div className="card-header-row">
            <div>
              <span className="card-kicker">Evaluation Results</span>
              <h3 className="card-title-sm">Holdout Test Set Performance (20%)</h3>
            </div>
            {savedToIndexedDb && (
              <span className="saved-indexeddb-badge">
                <CheckIcon size={14} /> Saved to IndexedDB
              </span>
            )}
          </div>

          {/* Side-by-Side Model Comparison */}
          <div className="models-comparison-grid">
            {/* k-NN Baseline Card */}
            <div className="model-score-card">
              <div className="score-card-top">
                <span className="score-model-title">k-Nearest Neighbors</span>
                <span className="score-model-badge">Baseline (k=5)</span>
              </div>
              <div className="score-hero">
                <span className="score-num">
                  {(evaluationResults.knnMetrics.accuracy * 100).toFixed(1)}%
                </span>
                <span className="score-label">Test Accuracy</span>
              </div>
              <div className="score-details-row">
                <div className="score-micro-item">
                  <span>Macro F1:</span>
                  <strong>{(evaluationResults.knnMetrics.macroAvg.f1 * 100).toFixed(1)}%</strong>
                </div>
                <div className="score-micro-item">
                  <span>Test Samples:</span>
                  <strong>{evaluationResults.testCount}</strong>
                </div>
              </div>
            </div>

            {/* Deep MLP Card */}
            <div className="model-score-card model-score-card-highlight">
              <div className="score-card-top">
                <span className="score-model-title">Multilayer Perceptron (MLP)</span>
                <span className="score-model-badge badge-mlp">TF.js Deep Learning</span>
              </div>
              <div className="score-hero">
                <span className="score-num score-num-highlight">
                  {(evaluationResults.mlpMetrics.accuracy * 100).toFixed(1)}%
                </span>
                <span className="score-label">Test Accuracy</span>
              </div>
              <div className="score-details-row">
                <div className="score-micro-item">
                  <span>Macro F1:</span>
                  <strong>{(evaluationResults.mlpMetrics.macroAvg.f1 * 100).toFixed(1)}%</strong>
                </div>
                <div className="score-micro-item">
                  <span>Optimizer:</span>
                  <strong>Adam (50 epochs)</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Per-Class Precision, Recall, F1 Table */}
          <div className="metrics-table-wrapper">
            <h4 className="subcard-title">Per-Class Classification Metrics (MLP)</h4>
            <table className="sianor-table">
              <thead>
                <tr>
                  <th>Sign Label</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support (Test Samples)</th>
                </tr>
              </thead>
              <tbody>
                {evaluationResults.mlpMetrics.classMetrics.map(cm => (
                  <tr key={cm.label}>
                    <td className="cell-label"><strong>{cm.label}</strong></td>
                    <td>{(cm.precision * 100).toFixed(1)}%</td>
                    <td>{(cm.recall * 100).toFixed(1)}%</td>
                    <td>{(cm.f1 * 100).toFixed(1)}%</td>
                    <td>{cm.support}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="tfoot-macro">
                  <td><strong>Macro Average</strong></td>
                  <td>{(evaluationResults.mlpMetrics.macroAvg.precision * 100).toFixed(1)}%</td>
                  <td>{(evaluationResults.mlpMetrics.macroAvg.recall * 100).toFixed(1)}%</td>
                  <td>{(evaluationResults.mlpMetrics.macroAvg.f1 * 100).toFixed(1)}%</td>
                  <td>{evaluationResults.testCount}</td>
                </tr>
              </tfoot>
            </table>
          </div>

          {/* Confusion Matrix Heatmap */}
          <div className="cm-section">
            <h4 className="subcard-title">Confusion Matrix Heatmap (Neural Network)</h4>
            <p className="card-instruction">
              Each row represents actual signs performed; each column represents the model's prediction.
              High diagonal values indicate robust, error-free classification.
            </p>
            <ConfusionMatrix
              matrix={evaluationResults.mlpMetrics.matrix}
              labels={evaluationResults.splitLabels}
            />
          </div>

          {/* Call to Action */}
          <div className="results-cta-bar">
            <span>Model is ready for live translation!</span>
            <button
              type="button"
              className="sianor-btn sianor-btn-primary"
              onClick={onGoToTranslate}
            >
              Test Live in Translate Tab →
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
