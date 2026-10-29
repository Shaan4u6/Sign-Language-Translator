/**
 * Sianor Preprocessing & Data Augmentation Pipeline
 * 
 * Preprocesses raw 21 MediaPipe 3D hand landmarks into an invariant 63-dimensional feature vector.
 * 
 * Pipeline:
 * 1. Wrist Translation: Translate all landmarks so landmark 0 (wrist) is at (0, 0, 0).
 * 2. Scale Normalization: Divide coordinates by the maximum Euclidean distance to any landmark.
 * 3. Handedness Reflection: Reflect the X-coordinate if the hand is Left, normalizing all hands to Right-handed representation.
 * 4. Vector Flattening: Flatten 21 * (x, y, z) into a 1D float vector of length 63.
 */

/**
 * Normalizes a single frame of 21 landmarks.
 * @param {Array<{x: number, y: number, z: number}>} landmarks - Raw landmarks from MediaPipe
 * @param {string|boolean} handedness - 'Left'/'Right' or boolean isLeft
 * @returns {Float32Array} - Normalized 63-element feature vector
 */
export function normalizeLandmarks(landmarks, handedness = 'Right') {
  if (!landmarks || landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  // Determine if left hand needs mirroring
  const isLeft = typeof handedness === 'string' 
    ? handedness.toLowerCase() === 'left' 
    : Boolean(handedness);

  // Step 1: Translate landmarks so wrist (landmark 0) is at origin (0, 0, 0)
  const wrist = landmarks[0];
  const translated = new Array(21);
  let maxDistance = 0;

  for (let i = 0; i < 21; i++) {
    const pt = landmarks[i];
    let tx = pt.x - wrist.x;
    const ty = pt.y - wrist.y;
    const tz = (pt.z !== undefined ? pt.z : 0) - (wrist.z !== undefined ? wrist.z : 0);

    // Step 3: Flip left hand to match right hand orientation
    if (isLeft) {
      tx = -tx;
    }

    translated[i] = { x: tx, y: ty, z: tz };

    // Step 2: Compute Euclidean distance from origin to find maximum distance
    const dist = Math.hypot(tx, ty, tz);
    if (dist > maxDistance) {
      maxDistance = dist;
    }
  }

  // Prevent division by zero if hand is somehow a single point
  const scale = maxDistance > 1e-6 ? maxDistance : 1.0;

  // Step 4: Scale-normalize and flatten to 63-element vector [x0, y0, z0, x1, y1, z1, ...]
  const vector = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const pt = translated[i];
    const idx = i * 3;
    vector[idx] = pt.x / scale;
    vector[idx + 1] = pt.y / scale;
    vector[idx + 2] = pt.z / scale;
  }

  return vector;
}

/**
 * Augments a normalized 63-element vector with random rotation, scaling, and Gaussian jitter.
 * @param {Float32Array|number[]} vector - Normalized 63-element vector
 * @param {Object} options - Augmentation hyperparameters
 * @returns {Float32Array} - Augmented 63-element vector
 */
export function augmentSample(vector, options = {}) {
  const {
    maxRotationDeg = 12,     // Rotation around Z axis (in-plane tilt)
    scaleJitter = 0.08,      // Random scale factor ±8%
    noiseStdDev = 0.015,     // Gaussian noise standard deviation
  } = options;

  const result = new Float32Array(63);

  // Random rotation angle in radians
  const angleDeg = (Math.random() * 2 - 1) * maxRotationDeg;
  const angleRad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  // Random scale factor [1 - jitter, 1 + jitter]
  const scale = 1 + (Math.random() * 2 - 1) * scaleJitter;

  for (let i = 0; i < 21; i++) {
    const idx = i * 3;
    const x = vector[idx];
    const y = vector[idx + 1];
    const z = vector[idx + 2];

    // 2D in-plane rotation around Z axis
    const rx = (x * cos - y * sin) * scale;
    const ry = (x * sin + y * cos) * scale;
    const rz = z * scale;

    // Box-Muller transform for Gaussian noise
    const u1 = Math.max(1e-7, Math.random());
    const u2 = Math.random();
    const gNoise1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * noiseStdDev;
    const gNoise2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2) * noiseStdDev;
    const gNoise3 = (Math.random() * 2 - 1) * noiseStdDev;

    result[idx] = rx + gNoise1;
    result[idx + 1] = ry + gNoise2;
    result[idx + 2] = rz + gNoise3;
  }

  return result;
}

/**
 * Generates an augmented dataset by applying transformations to each sample.
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 * @param {number} copiesPerSample - Number of synthetic variations per sample
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function augmentDataset(dataset, copiesPerSample = 2) {
  const augmented = [];

  for (const item of dataset) {
    // Keep original
    augmented.push({
      features: item.features instanceof Float32Array ? item.features : new Float32Array(item.features),
      label: item.label,
    });

    // Add augmented variants
    for (let c = 0; c < copiesPerSample; c++) {
      augmented.push({
        features: augmentSample(item.features),
        label: item.label,
      });
    }
  }

  return augmented;
}
