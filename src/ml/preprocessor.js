/**
 * Sianor Dual-Hand Preprocessing & Data Augmentation Pipeline
 * 
 * Features: 130-dimensional vector
 * - Slot 1 (indices 0..62):   Hand 1 normalized 21 landmarks (x, y, z)
 * - Slot 2 (indices 63..125):  Hand 2 normalized 21 landmarks (x, y, z) [zeros if 1 hand]
 * - Offset (indices 126..128): 3D offset between wrists divided by average hand scale [zeros if 1 hand]
 * - Flag   (index 129):        Two hands visible indicator (1.0 = both hands, 0.0 = single hand)
 * 
 * Normalization Rules:
 * 1. For each hand independently:
 *    a) Translate wrist (landmark 0) to origin (0, 0, 0)
 *    b) Scale by the maximum Euclidean distance from wrist to any landmark
 * 2. Spatial Hand Ordering:
 *    Hands are sorted by wrist screen X-position (leftmost first).
 *    Single hand is always placed in Slot 1; Slot 2 is filled with zeros.
 * 3. Inter-Hand Offset:
 *    Calculates 3D displacement (wrist2 - wrist1) scaled by average hand size.
 */

/**
 * Normalizes a single hand's 21 landmarks.
 * @param {Array<{x: number, y: number, z: number}>} landmarks
 * @returns {{ vector63: Float32Array, maxDist: number, wrist: {x: number, y: number, z: number} }}
 */
export function normalizeSingleHand(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  const wrist = landmarks[0];
  const translated = new Array(21);
  let maxDistance = 0;

  for (let i = 0; i < 21; i++) {
    const pt = landmarks[i];
    const tx = pt.x - wrist.x;
    const ty = pt.y - wrist.y;
    const tz = (pt.z !== undefined ? pt.z : 0) - (wrist.z !== undefined ? wrist.z : 0);

    translated[i] = { x: tx, y: ty, z: tz };

    const dist = Math.hypot(tx, ty, tz);
    if (dist > maxDistance) {
      maxDistance = dist;
    }
  }

  const scale = maxDistance > 1e-6 ? maxDistance : 1.0;
  const vector63 = new Float32Array(63);

  for (let i = 0; i < 21; i++) {
    const pt = translated[i];
    const idx = i * 3;
    vector63[idx] = pt.x / scale;
    vector63[idx + 1] = pt.y / scale;
    vector63[idx + 2] = pt.z / scale;
  }

  return {
    vector63,
    maxDist: scale,
    wrist: {
      x: wrist.x,
      y: wrist.y,
      z: wrist.z !== undefined ? wrist.z : 0,
    },
  };
}

/**
 * Compiles a 130-feature vector from one or two hands.
 * @param {Array<Array<{x: number, y: number, z: number}>> | Array<{x: number, y: number, z: number}>} handsInput
 * @returns {Float32Array} - Normalized 130-element feature vector
 */
export function normalizeLandmarks(handsInput) {
  if (!handsInput) return null;

  // Standardize input to array of hand landmark arrays
  const handsList = Array.isArray(handsInput[0])
    ? handsInput.filter(h => h && h.length === 21)
    : (handsInput.length === 21 && handsInput[0].x !== undefined ? [handsInput] : []);

  if (handsList.length === 0) {
    return null;
  }

  const vector = new Float32Array(130);

  if (handsList.length === 1) {
    // Single Hand Gesture: place in Slot 1, Slot 2 and offset remain zero
    const { vector63 } = normalizeSingleHand(handsList[0]);
    vector.set(vector63, 0);
    // vector[63..125] are 0.0
    // vector[126..128] are 0.0 (offset)
    // vector[129] is 0.0 (single hand flag)
    return vector;
  }

  // Two Hands Gesture: order by wrist screen X-position (leftmost first)
  let hand1 = handsList[0];
  let hand2 = handsList[1];

  if (hand1[0].x > hand2[0].x) {
    hand1 = handsList[1];
    hand2 = handsList[0];
  }

  const norm1 = normalizeSingleHand(hand1);
  const norm2 = normalizeSingleHand(hand2);

  // Slot 1 (indices 0..62)
  vector.set(norm1.vector63, 0);

  // Slot 2 (indices 63..125)
  vector.set(norm2.vector63, 63);

  // 3D offset between the two wrists divided by average hand size
  const avgScale = (norm1.maxDist + norm2.maxDist) / 2.0;
  const safeAvgScale = avgScale > 1e-6 ? avgScale : 1.0;

  const dx = (norm2.wrist.x - norm1.wrist.x) / safeAvgScale;
  const dy = (norm2.wrist.y - norm1.wrist.y) / safeAvgScale;
  const dz = (norm2.wrist.z - norm1.wrist.z) / safeAvgScale;

  vector[126] = dx;
  vector[127] = dy;
  vector[128] = dz;
  vector[129] = 1.0; // Two hands visible flag

  return vector;
}

/**
 * Augments a 130-element vector with rotation, scaling, Gaussian noise, and horizontal mirroring.
 * @param {Float32Array|number[]} vector
 * @param {Object} options
 * @returns {Float32Array}
 */
export function augmentSample(vector, options = {}) {
  const {
    maxRotationDeg = 12,
    scaleJitter = 0.08,
    noiseStdDev = 0.015,
    allowMirror = true,
  } = options;

  const result = new Float32Array(130);
  const isTwoHands = vector[129] > 0.5;

  // Decide if this augmented copy should be horizontally mirrored
  const shouldMirror = allowMirror && Math.random() < 0.5;

  // Random in-plane rotation angle around Z-axis
  const angleDeg = (Math.random() * 2 - 1) * maxRotationDeg;
  const angleRad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  // Random scale factor
  const scale = 1 + (Math.random() * 2 - 1) * scaleJitter;

  // Helper to transform a 63-element hand slot
  const transformSlot = (srcOffset, destOffset, invertX) => {
    for (let i = 0; i < 21; i++) {
      const idx = srcOffset + i * 3;
      const destIdx = destOffset + i * 3;

      let x = vector[idx];
      let y = vector[idx + 1];
      let z = vector[idx + 2];

      if (invertX) {
        x = -x;
      }

      // Rotate and scale
      const rx = (x * cos - y * sin) * scale;
      const ry = (x * sin + y * cos) * scale;
      const rz = z * scale;

      // Gaussian noise
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      const n1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * noiseStdDev;
      const n2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2) * noiseStdDev;
      const n3 = (Math.random() * 2 - 1) * noiseStdDev;

      result[destIdx] = rx + n1;
      result[destIdx + 1] = ry + n2;
      result[destIdx + 2] = rz + n3;
    }
  };

  if (isTwoHands) {
    if (shouldMirror) {
      // Swapping slots: leftmost hand becomes rightmost and vice-versa
      transformSlot(0, 63, true);  // Slot 1 -> Slot 2 (mirrored)
      transformSlot(63, 0, true);  // Slot 2 -> Slot 1 (mirrored)

      // Mirror inter-wrist offset (dx is negated)
      let dx = -vector[126];
      let dy = vector[127];
      let dz = vector[128];

      result[126] = (dx * cos - dy * sin) * scale;
      result[127] = (dx * sin + dy * cos) * scale;
      result[128] = dz * scale;
    } else {
      transformSlot(0, 0, false);
      transformSlot(63, 63, false);

      let dx = vector[126];
      let dy = vector[127];
      let dz = vector[128];

      result[126] = (dx * cos - dy * sin) * scale;
      result[127] = (dx * sin + dy * cos) * scale;
      result[128] = dz * scale;
    }
    result[129] = 1.0;
  } else {
    // Single hand: always stays in Slot 1
    transformSlot(0, 0, shouldMirror);
    // Slot 2 and offset remain zero
    result[129] = 0.0;
  }

  return result;
}

/**
 * Generates an augmented dataset.
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 * @param {number} copiesPerSample
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function augmentDataset(dataset, copiesPerSample = 2) {
  const augmented = [];

  for (const item of dataset) {
    augmented.push({
      features: item.features instanceof Float32Array ? item.features : new Float32Array(item.features),
      label: item.label,
    });

    for (let c = 0; c < copiesPerSample; c++) {
      augmented.push({
        features: augmentSample(item.features),
        label: item.label,
      });
    }
  }

  return augmented;
}
