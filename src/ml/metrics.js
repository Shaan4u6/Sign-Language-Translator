/**
 * Machine Learning Evaluation & Dataset Metrics
 * 
 * Provides:
 * - Stratified 80/20 Train/Test split preserving class distribution.
 * - Accuracy score calculation.
 * - Per-class Precision, Recall, F1-Score, and Support.
 * - N x N Confusion Matrix for error analysis and heatmap rendering.
 */

/**
 * Performs a stratified train/test split.
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 * @param {number} trainRatio - e.g. 0.8 for 80/20 split
 * @returns {{
 *   X_train: Float32Array[],
 *   y_train: number[],
 *   X_test: Float32Array[],
 *   y_test: number[],
 *   labels: string[]
 * }}
 */
export function stratifiedSplit(dataset, trainRatio = 0.8) {
  if (!dataset || dataset.length === 0) {
    throw new Error('Dataset is empty');
  }

  // Extract unique labels
  const labelSet = new Set();
  for (const item of dataset) {
    labelSet.add(item.label);
  }
  const labels = Array.from(labelSet).sort();
  const labelToIndex = new Map(labels.map((l, i) => [l, i]));

  // Group samples by class label
  const grouped = new Map();
  for (const label of labels) {
    grouped.set(label, []);
  }
  for (const item of dataset) {
    grouped.get(item.label).push(item);
  }

  const X_train = [];
  const y_train = [];
  const X_test = [];
  const y_test = [];

  for (const [label, samples] of grouped.entries()) {
    const classIdx = labelToIndex.get(label);

    // Shuffle class samples using Fisher-Yates
    const shuffled = [...samples];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    // Split count: ensure at least 1 test sample if group >= 2
    let trainCount = Math.floor(shuffled.length * trainRatio);
    if (shuffled.length >= 2 && trainCount === shuffled.length) {
      trainCount = shuffled.length - 1;
    }
    if (trainCount === 0 && shuffled.length > 0) {
      trainCount = 1;
    }

    const trainGroup = shuffled.slice(0, trainCount);
    const testGroup = shuffled.slice(trainCount);

    for (const item of trainGroup) {
      X_train.push(item.features instanceof Float32Array ? item.features : new Float32Array(item.features));
      y_train.push(classIdx);
    }
    for (const item of testGroup) {
      X_test.push(item.features instanceof Float32Array ? item.features : new Float32Array(item.features));
      y_test.push(classIdx);
    }
  }

  return { X_train, y_train, X_test, y_test, labels };
}

/**
 * Calculates raw accuracy (0.0 to 1.0).
 * @param {number[]} y_true
 * @param {number[]} y_pred
 * @returns {number}
 */
export function calculateAccuracy(y_true, y_pred) {
  if (y_true.length === 0 || y_true.length !== y_pred.length) return 0;
  let correct = 0;
  for (let i = 0; i < y_true.length; i++) {
    if (y_true[i] === y_pred[i]) correct++;
  }
  return correct / y_true.length;
}

/**
 * Computes confusion matrix and per-class performance metrics.
 * @param {number[]} y_true - Ground truth class indices
 * @param {number[]} y_pred - Predicted class indices
 * @param {string[]} labels - Class names
 * @returns {{
 *   matrix: number[][],
 *   classMetrics: Array<{
 *     label: string,
 *     precision: number,
 *     recall: number,
 *     f1: number,
 *     support: number
 *   }>,
 *   macroAvg: { precision: number, recall: number, f1: number },
 *   weightedAvg: { precision: number, recall: number, f1: number },
 *   accuracy: number
 * }}
 */
export function computeEvaluationMetrics(y_true, y_pred, labels) {
  const K = labels.length;

  // Initialize K x K matrix with zeros
  // matrix[actual][predicted]
  const matrix = Array.from({ length: K }, () => new Array(K).fill(0));

  for (let i = 0; i < y_true.length; i++) {
    const trueClass = y_true[i];
    const predClass = y_pred[i];
    if (trueClass >= 0 && trueClass < K && predClass >= 0 && predClass < K) {
      matrix[trueClass][predClass]++;
    }
  }

  const classMetrics = [];
  let totalSupport = 0;
  let sumPrecision = 0;
  let sumRecall = 0;
  let sumF1 = 0;
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;

  for (let c = 0; c < K; c++) {
    const tp = matrix[c][c];

    // False Positives: other classes predicted as c
    let fp = 0;
    for (let r = 0; r < K; r++) {
      if (r !== c) fp += matrix[r][c];
    }

    // False Negatives: actual c predicted as other classes
    let fn = 0;
    for (let col = 0; col < K; col++) {
      if (col !== c) fn += matrix[c][col];
    }

    const support = tp + fn;
    totalSupport += support;

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    classMetrics.push({
      label: labels[c],
      precision,
      recall,
      f1,
      support,
    });

    sumPrecision += precision;
    sumRecall += recall;
    sumF1 += f1;
    weightedPrecision += precision * support;
    weightedRecall += recall * support;
    weightedF1 += f1 * support;
  }

  const macroAvg = {
    precision: K > 0 ? sumPrecision / K : 0,
    recall: K > 0 ? sumRecall / K : 0,
    f1: K > 0 ? sumF1 / K : 0,
  };

  const weightedAvg = {
    precision: totalSupport > 0 ? weightedPrecision / totalSupport : 0,
    recall: totalSupport > 0 ? weightedRecall / totalSupport : 0,
    f1: totalSupport > 0 ? weightedF1 / totalSupport : 0,
  };

  const accuracy = calculateAccuracy(y_true, y_pred);

  return {
    matrix,
    classMetrics,
    macroAvg,
    weightedAvg,
    accuracy,
  };
}
