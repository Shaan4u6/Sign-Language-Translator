/**
 * Sianor TensorFlow.js Multilayer Perceptron (MLP) Classifier
 * 
 * Deep learning classifier running 100% client-side in the browser via WebGL/WASM.
 * 
 * Architecture:
 * - Input: 63 features (21 hand landmarks * 3 coordinates [x, y, z])
 * - Hidden Layer 1: Dense 128 neurons, ReLU activation
 * - Regularization 1: Dropout (rate 0.30) to prevent overfitting on small gesture datasets
 * - Hidden Layer 2: Dense 64 neurons, ReLU activation
 * - Output Layer: Dense N neurons (number of gesture classes), Softmax activation
 * 
 * Training:
 * - Optimizer: Adam (adaptive learning rate, initial lr=0.001)
 * - Loss: Categorical Crossentropy (standard for multi-class probability outputs)
 * - Metrics: Accuracy
 */

import * as tf from '@tensorflow/tfjs';

export class TFHandModel {
  constructor() {
    this.model = null;
    this.labels = [];
    this.isTrained = false;
  }

  /**
   * Builds the MLP architecture.
   * @param {number} numClasses - Number of distinct sign labels
   */
  buildModel(numClasses) {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    const model = tf.sequential();

    // Layer 1: Dense 128 + ReLU
    model.add(tf.layers.dense({
      inputShape: [63],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
    }));

    // Dropout 0.3 for regularization
    model.add(tf.layers.dropout({ rate: 0.3 }));

    // Layer 2: Dense 64 + ReLU
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
    }));

    // Output: Softmax over classes
    model.add(tf.layers.dense({
      units: Math.max(1, numClasses),
      activation: 'softmax',
    }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.model = model;
    return model;
  }

  /**
   * Trains the model on provided features and targets.
   * @param {Array<Float32Array|number[]>} X_train
   * @param {number[]} y_train
   * @param {Array<Float32Array|number[]>} X_val
   * @param {number[]} y_val
   * @param {string[]} labels
   * @param {Object} options
   * @param {Function} options.onEpochEnd - Callback for live loss/acc charting
   * @param {number} options.epochs - Default 50
   * @param {number} options.batchSize - Default 16
   */
  async fit(X_train, y_train, X_val, y_val, labels, options = {}) {
    const {
      epochs = 50,
      batchSize = 16,
      onEpochEnd = null,
    } = options;

    this.labels = [...labels];
    const numClasses = labels.length;

    this.buildModel(numClasses);

    // Convert datasets to TF tensors
    const trainCount = X_train.length;
    const flatTrainX = new Float32Array(trainCount * 63);
    for (let i = 0; i < trainCount; i++) {
      flatTrainX.set(X_train[i], i * 63);
    }

    const valCount = X_val.length;
    const flatValX = new Float32Array(valCount * 63);
    for (let i = 0; i < valCount; i++) {
      flatValX.set(X_val[i], i * 63);
    }

    const xs = tf.tensor2d(flatTrainX, [trainCount, 63]);
    const ys = tf.oneHot(tf.tensor1d(y_train, 'int32'), numClasses);

    const valXs = tf.tensor2d(flatValX, [valCount, 63]);
    const valYs = tf.oneHot(tf.tensor1d(y_val, 'int32'), numClasses);

    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        batchSize,
        validationData: [valXs, valYs],
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (onEpochEnd) {
              onEpochEnd({
                epoch: epoch + 1,
                loss: logs.loss,
                accuracy: logs.acc || logs.accuracy,
                valLoss: logs.val_loss,
                valAccuracy: logs.val_acc || logs.val_accuracy,
              });
            }
          },
        },
      });

      this.isTrained = true;
      return history;
    } finally {
      // Free tensor memory immediately
      xs.dispose();
      ys.dispose();
      valXs.dispose();
      valYs.dispose();
    }
  }

  /**
   * Predicts class distribution for a single normalized 63-element vector.
   * @param {Float32Array|number[]} vector
   * @returns {{
   *   classIndex: number,
   *   label: string,
   *   confidence: number,
   *   allConfidences: Record<string, number>
   * }}
   */
  predict(vector) {
    if (!this.model || !this.isTrained || this.labels.length === 0) {
      return { classIndex: -1, label: 'Untrained', confidence: 0, allConfidences: {} };
    }

    return tf.tidy(() => {
      const input = tf.tensor2d(vector, [1, 63]);
      const prediction = this.model.predict(input);
      const probabilities = prediction.dataSync();

      let maxProb = -1;
      let bestIndex = 0;

      const allConfidences = {};
      for (let i = 0; i < probabilities.length; i++) {
        const prob = probabilities[i];
        const labelName = this.labels[i] || `Class ${i}`;
        allConfidences[labelName] = prob;
        if (prob > maxProb) {
          maxProb = prob;
          bestIndex = i;
        }
      }

      return {
        classIndex: bestIndex,
        label: this.labels[bestIndex] || 'Unknown',
        confidence: maxProb,
        allConfidences,
      };
    });
  }

  /**
   * Predicts an array of test vectors.
   * @param {Array<Float32Array|number[]>} X_test
   * @returns {number[]} Array of predicted class indices
   */
  predictAll(X_test) {
    if (!this.model || !this.isTrained) return [];

    return tf.tidy(() => {
      const count = X_test.length;
      const flatX = new Float32Array(count * 63);
      for (let i = 0; i < count; i++) {
        flatX.set(X_test[i], i * 63);
      }
      const xs = tf.tensor2d(flatX, [count, 63]);
      const preds = this.model.predict(xs);
      const argmax = preds.argMax(-1);
      return Array.from(argmax.dataSync());
    });
  }

  /**
   * Saves model and label metadata to browser's IndexedDB.
   */
  async saveToIndexedDB(saveKey = 'sianor-hand-mlp') {
    if (!this.model || !this.isTrained) {
      throw new Error('No trained model available to save');
    }
    await this.model.save(`indexeddb://${saveKey}`);
    localStorage.setItem(`${saveKey}-labels`, JSON.stringify(this.labels));
  }

  /**
   * Loads saved model and label metadata from IndexedDB if present.
   */
  async loadFromIndexedDB(saveKey = 'sianor-hand-mlp') {
    try {
      const savedLabels = localStorage.getItem(`${saveKey}-labels`);
      if (!savedLabels) return false;

      const labels = JSON.parse(savedLabels);
      const loadedModel = await tf.loadLayersModel(`indexeddb://${saveKey}`);
      if (this.model) {
        this.model.dispose();
      }
      this.model = loadedModel;
      this.labels = labels;
      this.isTrained = true;
      return true;
    } catch (err) {
      console.warn('Could not load saved model from IndexedDB:', err);
      return false;
    }
  }

  /**
   * Disposes model tensors from memory.
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isTrained = false;
    }
  }
}
