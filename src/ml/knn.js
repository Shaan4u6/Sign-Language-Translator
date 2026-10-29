/**
 * Manual k-Nearest Neighbors (k-NN) Classifier
 * 
 * Implemented from first principles without external ML libraries.
 * Serves as an interpretable baseline classifier to compare against the Neural Network.
 * 
 * Algorithm:
 * 1. Store training feature vectors and corresponding class indices.
 * 2. For an input query vector, calculate Euclidean distance d(u, v) = sqrt(sum((u_i - v_i)^2)).
 * 3. Select the k training instances with smallest distances.
 * 4. Perform majority voting among the k neighbors.
 * 5. Compute confidence as (votes for winning class / k).
 */

export class KNNClassifier {
  /**
   * @param {number} k - Number of nearest neighbors to consider (default k=5)
   */
  constructor(k = 5) {
    this.k = k;
    this.trainX = [];      // Array of Float32Array features
    this.trainY = [];      // Array of class indices (0..N-1)
    this.labels = [];      // Array of label names corresponding to indices
    this.isTrained = false;
  }

  /**
   * Trains (indexes) the k-NN model.
   * @param {Array<Float32Array|number[]>} X - 63-element feature vectors
   * @param {number[]} y - Target class indices
   * @param {string[]} labels - List of unique class label names
   */
  fit(X, y, labels) {
    if (!X || X.length === 0) {
      throw new Error('Cannot fit k-NN with empty training dataset');
    }
    this.trainX = X.map(vec => (vec instanceof Float32Array ? vec : new Float32Array(vec)));
    this.trainY = [...y];
    this.labels = [...labels];
    this.isTrained = true;
  }

  /**
   * Calculates Euclidean distance between two 63-feature vectors.
   * @param {Float32Array|number[]} a
   * @param {Float32Array|number[]} b
   * @returns {number}
   */
  euclideanDistance(a, b) {
    let sumSq = 0;
    const len = a.length;
    for (let i = 0; i < len; i++) {
      const diff = a[i] - b[i];
      sumSq += diff * diff;
    }
    return Math.sqrt(sumSq);
  }

  /**
   * Predicts class and confidence for a single feature vector.
   * @param {Float32Array|number[]} vector - Normalized 63-element feature vector
   * @returns {{
   *   classIndex: number,
   *   label: string,
   *   confidence: number,
   *   voteCount: number,
   *   k: number,
   *   allConfidences: Record<string, number>
   * }}
   */
  predict(vector) {
    if (!this.isTrained || this.trainX.length === 0) {
      return { classIndex: -1, label: 'Untrained', confidence: 0, voteCount: 0, k: this.k, allConfidences: {} };
    }

    const n = this.trainX.length;
    const effectiveK = Math.min(this.k, n);

    // Compute distance to each training point
    const distances = new Array(n);
    for (let i = 0; i < n; i++) {
      distances[i] = {
        dist: this.euclideanDistance(vector, this.trainX[i]),
        classIndex: this.trainY[i],
      };
    }

    // Sort ascending by distance (find k nearest)
    distances.sort((a, b) => a.dist - b.dist);

    // Tally votes from the top-k nearest neighbors
    const voteCounts = {};
    for (let c = 0; c < this.labels.length; c++) {
      voteCounts[c] = 0;
    }

    for (let i = 0; i < effectiveK; i++) {
      const c = distances[i].classIndex;
      voteCounts[c] = (voteCounts[c] || 0) + 1;
    }

    // Find class with maximum votes
    let bestClass = 0;
    let maxVotes = -1;
    for (let c = 0; c < this.labels.length; c++) {
      const count = voteCounts[c] || 0;
      if (count > maxVotes) {
        maxVotes = count;
        bestClass = c;
      }
    }

    const confidence = maxVotes / effectiveK;

    const allConfidences = {};
    for (let c = 0; c < this.labels.length; c++) {
      const name = this.labels[c] || `Class ${c}`;
      allConfidences[name] = (voteCounts[c] || 0) / effectiveK;
    }

    return {
      classIndex: bestClass,
      label: this.labels[bestClass] || `Unknown`,
      confidence: confidence,
      voteCount: maxVotes,
      k: effectiveK,
      allConfidences,
    };
  }

  /**
   * Predicts an array of test vectors.
   * @param {Array<Float32Array|number[]>} X_test
   * @returns {number[]} Array of predicted class indices
   */
  predictAll(X_test) {
    return X_test.map(vec => this.predict(vec).classIndex);
  }
}
