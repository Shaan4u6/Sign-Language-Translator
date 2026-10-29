/**
 * Sianor TensorFlow.js Multilayer Perceptron (MLP) Classifier
 * 
 * Deep learning classifier running client-side in the browser via WebGL/WASM.
 * Supports both one-hand and two-hand sign language recognition.
 * 
 * Architecture:
 * - Input: 130 features (Hand 1: 63, Hand 2: 63, Inter-wrist 3D offset: 3, Two-hands flag: 1)
 * - Hidden Layer 1: Dense 128 neurons, ReLU activation
 * - Regularization 1: Dropout (rate 0.30)
 * - Hidden Layer 2: Dense 64 neurons, ReLU activation
 * - Output Layer: Dense N neurons (number of gesture classes), Softmax activation
 * 
 * Training:
 * - Optimizer: Adam (adaptive learning rate, initial lr=0.001)
 * - Loss: Categorical Crossentropy
 * - Metrics: Accuracy
 */

import * as tf from '@tensorflow/tfjs';

export const FEATURE_SIZE = 130;

export class TFHandModel {
  constructor() {
    this.model = null;
    this.labels = [];
    this.isTrained = false;
  }

  /**
   * Builds the MLP architecture with 130 inputs.
   * @param {number} numClasses - Number of distinct sign labels
   */
  buildModel(numClasses) {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    const model = tf.sequential();

    // Layer 1: Dense 128 + ReLU (Input shape: 130)
    model.add(tf.layers.dense({
      inputShape: [FEATURE_SIZE],
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
   * Trains the model on 130-element features and targets.
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

    // Convert datasets to TF tensors (130 features per sample)
    const trainCount = X_train.length;
    const flatTrainX = new Float32Array(trainCount * FEATURE_SIZE);
    for (let i = 0; i < trainCount; i++) {
      flatTrainX.set(X_train[i], i * FEATURE_SIZE);
    }

    const valCount = X_val.length;
    const flatValX = new Float32Array(valCount * FEATURE_SIZE);
    for (let i = 0; i < valCount; i++) {
      flatValX.set(X_val[i], i * FEATURE_SIZE);
    }

    const xs = tf.tensor2d(flatTrainX, [trainCount, FEATURE_SIZE]);
    const ys = tf.oneHot(tf.tensor1d(y_train, 'int32'), numClasses);

    const valXs = tf.tensor2d(flatValX, [valCount, FEATURE_SIZE]);
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
      xs.dispose();
      ys.dispose();
      valXs.dispose();
      valYs.dispose();
    }
  }

  /**
   * Predicts class distribution for a single normalized 130-element vector.
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
      const input = tf.tensor2d(vector, [1, FEATURE_SIZE]);
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
      const flatX = new Float32Array(count * FEATURE_SIZE);
      for (let i = 0; i < count; i++) {
        flatX.set(X_test[i], i * FEATURE_SIZE);
      }
      const xs = tf.tensor2d(flatX, [count, FEATURE_SIZE]);
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
    localStorage.setItem(`${saveKey}-features`, String(FEATURE_SIZE));
  }

  /**
   * Loads saved model and label metadata from IndexedDB if present.
   * Clears old model if feature size does not match FEATURE_SIZE (130).
   * @returns {Promise<boolean | { legacyCleared: boolean }>}
   */
  async loadFromIndexedDB(saveKey = 'sianor-hand-mlp') {
    try {
      const savedLabels = localStorage.getItem(`${saveKey}-labels`);
      if (!savedLabels) return false;

      const labels = JSON.parse(savedLabels);
      const loadedModel = await tf.loadLayersModel(`indexeddb://${saveKey}`);

      // Check input shape compatibility
      const inputDim = loadedModel.inputs && loadedModel.inputs[0]?.shape?.[1];
      if (inputDim !== FEATURE_SIZE) {
        console.warn(`Saved model has input shape ${inputDim}, but dual-hand requires ${FEATURE_SIZE}. Clearing legacy model...`);
        loadedModel.dispose();
        try {
          await tf.io.removeModel(`indexeddb://${saveKey}`);
        } catch {
          // ignore
        }
        localStorage.removeItem(`${saveKey}-labels`);
        localStorage.removeItem(`${saveKey}-features`);
        return { legacyCleared: true };
      }

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
   * Exports TF.js model files (.json topology and .weights.bin binary) to user's downloads folder.
   * Also downloads a companion labels.json file.
   * @param {string} baseName
   */
  async exportModelFiles(baseName = 'sianor-hand-mlp') {
    if (!this.model || !this.isTrained) {
      throw new Error('No trained model available to export');
    }

    // Embed user-defined metadata into model topology
    this.model.userDefinedMetadata = {
      labels: this.labels,
      numClasses: this.labels.length,
      featureSize: FEATURE_SIZE,
      exportedAt: new Date().toISOString(),
      architecture: '130 -> 128(relu) -> drop(0.3) -> 64(relu) -> softmax',
    };

    // 1. Download model.json and model.weights.bin via TF.js downloads IO handler
    await this.model.save(`downloads://${baseName}`);

    // 2. Also download companion labels JSON file
    try {
      const labelsBlob = new Blob([
        JSON.stringify({ labels: this.labels, numClasses: this.labels.length, featureSize: FEATURE_SIZE }, null, 2)
      ], { type: 'application/json' });
      const url = URL.createObjectURL(labelsBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${baseName}.labels.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (lblErr) {
      console.warn('Could not export companion labels file:', lblErr);
    }
  }

  /**
   * Loads a TF.js model from user-selected files (.json + .bin + optional .labels.json).
   * @param {FileList|File[]} files
   * @param {string} saveKey
   * @returns {Promise<{ model: tf.LayersModel, labels: string[] }>}
   */
  async loadModelFromFiles(files, saveKey = 'sianor-hand-mlp') {
    if (!files || files.length === 0) {
      throw new Error('No files provided');
    }

    const fileList = Array.from(files);
    const jsonFile = fileList.find(f => f.name.endsWith('.json') && !f.name.includes('labels'));
    const binFile = fileList.find(f => f.name.endsWith('.bin') || f.name.endsWith('.weights.bin'));
    const labelFile = fileList.find(f => f.name.includes('label') && f.name.endsWith('.json'));

    if (!jsonFile) {
      throw new Error('Missing model architecture file (.json)');
    }
    if (!binFile) {
      throw new Error('Missing weights file (.bin or .weights.bin)');
    }

    // Load layers model using tf.io.browserFiles
    const loadedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));

    const inputDim = loadedModel.inputs && loadedModel.inputs[0]?.shape?.[1];
    if (inputDim !== FEATURE_SIZE) {
      loadedModel.dispose();
      throw new Error(`Incompatible model input size (${inputDim} features). Sianor Dual-Hand expects ${FEATURE_SIZE} features.`);
    }

    loadedModel.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    let detectedLabels = [];

    // 1. Check userDefinedMetadata inside loadedModel
    if (loadedModel.userDefinedMetadata && Array.isArray(loadedModel.userDefinedMetadata.labels)) {
      detectedLabels = loadedModel.userDefinedMetadata.labels;
    }

    // 2. Check companion label file if provided
    if (detectedLabels.length === 0 && labelFile) {
      try {
        const text = await labelFile.text();
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) detectedLabels = parsed;
        else if (parsed.labels && Array.isArray(parsed.labels)) detectedLabels = parsed.labels;
      } catch (e) {
        console.warn('Could not parse companion label file:', e);
      }
    }

    // 3. Check raw JSON file text for userDefinedMetadata
    if (detectedLabels.length === 0) {
      try {
        const jsonText = await jsonFile.text();
        const parsedJson = JSON.parse(jsonText);
        if (parsedJson.userDefinedMetadata && Array.isArray(parsedJson.userDefinedMetadata.labels)) {
          detectedLabels = parsedJson.userDefinedMetadata.labels;
        }
      } catch (e) {
        console.warn('Could not parse userDefinedMetadata from json text:', e);
      }
    }

    if (this.model) {
      this.model.dispose();
    }

    this.model = loadedModel;
    this.labels = detectedLabels;
    this.isTrained = true;

    // Cache to IndexedDB
    try {
      await this.saveToIndexedDB(saveKey);
    } catch (saveErr) {
      console.warn('Could not cache loaded model to IndexedDB:', saveErr);
    }

    return { model: loadedModel, labels: detectedLabels };
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
