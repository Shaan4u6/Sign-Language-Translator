/**
 * Sianor Pre-recorded / Synthesized Realistic Dual-Hand Demo Dataset (130 Features)
 * 
 * Provides calibrated anatomical poses for the 8 standard signs, supporting both
 * one-hand and two-hand sign language:
 * 1. Hello (1 Hand): Open hand, fingers spread upward
 * 2. Thank You (2 Hands): Both hands flat, angled forward in appreciation
 * 3. Yes (1 Hand): Closed fist nodding pose
 * 4. No (1 Hand): Index and middle finger pinch to thumb
 * 5. I Love You (1 Hand): ASL ILY (thumb, index, pinky extended)
 * 6. Stop (1 Hand): Open palm vertical facing camera
 * 7. Please (2 Hands): Both flat open hands placed in gentle prayer / circular chest pose
 * 8. OK (1 Hand): Thumb and index tips touching in an "O", 3 fingers upright
 */

import { normalizeLandmarks, augmentSample } from './preprocessor.js';

/**
 * Generates base 21 anatomical landmarks for a hand centered at an arbitrary wrist anchor.
 */
function getSingleHandLandmarks(shapeType, wristAnchor = { x: 0.5, y: 0.82, z: 0.0 }, isLeftHand = false) {
  const xSign = isLeftHand ? -1 : 1;
  const pts = new Array(21);

  // Wrist
  pts[0] = { ...wristAnchor };

  // Knuckle anchors relative to wrist
  pts[1] = { x: wristAnchor.x - 0.08 * xSign, y: wristAnchor.y - 0.08, z: -0.02 };
  pts[2] = { x: wristAnchor.x - 0.14 * xSign, y: wristAnchor.y - 0.16, z: -0.04 };
  pts[5] = { x: wristAnchor.x - 0.06 * xSign, y: wristAnchor.y - 0.26, z: -0.02 };
  pts[9] = { x: wristAnchor.x, y: wristAnchor.y - 0.28, z: -0.01 };
  pts[13] = { x: wristAnchor.x + 0.06 * xSign, y: wristAnchor.y - 0.26, z: 0.0 };
  pts[17] = { x: wristAnchor.x + 0.12 * xSign, y: wristAnchor.y - 0.22, z: 0.01 };

  const extendFinger = (mcpIdx, pipIdx, dipIdx, tipIdx, targetTipX, targetTipY, targetTipZ) => {
    const mcp = pts[mcpIdx];
    pts[pipIdx] = {
      x: mcp.x + (targetTipX - mcp.x) * 0.38,
      y: mcp.y + (targetTipY - mcp.y) * 0.38,
      z: mcp.z + (targetTipZ - mcp.z) * 0.38,
    };
    pts[dipIdx] = {
      x: mcp.x + (targetTipX - mcp.x) * 0.72,
      y: mcp.y + (targetTipY - mcp.y) * 0.72,
      z: mcp.z + (targetTipZ - mcp.z) * 0.72,
    };
    pts[tipIdx] = { x: targetTipX, y: targetTipY, z: targetTipZ };
  };

  const curlFinger = (mcpIdx, pipIdx, dipIdx, tipIdx, inwardX = 0, inwardY = 0.08) => {
    const mcp = pts[mcpIdx];
    pts[pipIdx] = { x: mcp.x + inwardX * 0.5, y: mcp.y - 0.06, z: -0.08 };
    pts[dipIdx] = { x: mcp.x + inwardX, y: mcp.y + inwardY * 0.6, z: -0.05 };
    pts[tipIdx] = { x: mcp.x + inwardX * 0.8, y: mcp.y + inwardY, z: -0.02 };
  };

  switch (shapeType) {
    case 'open': {
      pts[3] = { x: wristAnchor.x - 0.20 * xSign, y: wristAnchor.y - 0.22, z: -0.05 };
      pts[4] = { x: wristAnchor.x - 0.26 * xSign, y: wristAnchor.y - 0.28, z: -0.06 };
      extendFinger(5, 6, 7, 8, wristAnchor.x - 0.12 * xSign, wristAnchor.y - 0.54, -0.03);
      extendFinger(9, 10, 11, 12, wristAnchor.x, wristAnchor.y - 0.58, -0.02);
      extendFinger(13, 14, 15, 16, wristAnchor.x + 0.11 * xSign, wristAnchor.y - 0.54, -0.01);
      extendFinger(17, 18, 19, 20, wristAnchor.x + 0.22 * xSign, wristAnchor.y - 0.47, 0.0);
      break;
    }
    case 'flat_forward': {
      pts[3] = { x: wristAnchor.x - 0.14 * xSign, y: wristAnchor.y - 0.22, z: -0.06 };
      pts[4] = { x: wristAnchor.x - 0.18 * xSign, y: wristAnchor.y - 0.28, z: -0.08 };
      extendFinger(5, 6, 7, 8, wristAnchor.x - 0.06 * xSign, wristAnchor.y - 0.48, -0.15);
      extendFinger(9, 10, 11, 12, wristAnchor.x, wristAnchor.y - 0.50, -0.15);
      extendFinger(13, 14, 15, 16, wristAnchor.x + 0.06 * xSign, wristAnchor.y - 0.47, -0.14);
      extendFinger(17, 18, 19, 20, wristAnchor.x + 0.13 * xSign, wristAnchor.y - 0.42, -0.12);
      break;
    }
    case 'fist': {
      pts[3] = { x: wristAnchor.x - 0.10 * xSign, y: wristAnchor.y - 0.21, z: -0.08 };
      pts[4] = { x: wristAnchor.x - 0.05 * xSign, y: wristAnchor.y - 0.24, z: -0.10 };
      curlFinger(5, 6, 7, 8, -0.02 * xSign, 0.08);
      curlFinger(9, 10, 11, 12, 0.0, 0.08);
      curlFinger(13, 14, 15, 16, 0.02 * xSign, 0.08);
      curlFinger(17, 18, 19, 20, 0.03 * xSign, 0.07);
      break;
    }
    case 'pinch_no': {
      pts[3] = { x: wristAnchor.x - 0.06 * xSign, y: wristAnchor.y - 0.29, z: -0.07 };
      pts[4] = { x: wristAnchor.x - 0.02 * xSign, y: wristAnchor.y - 0.32, z: -0.08 };
      pts[6] = { x: wristAnchor.x - 0.07 * xSign, y: wristAnchor.y - 0.34, z: -0.06 };
      pts[7] = { x: wristAnchor.x - 0.04 * xSign, y: wristAnchor.y - 0.33, z: -0.07 };
      pts[8] = { x: wristAnchor.x - 0.02 * xSign, y: wristAnchor.y - 0.32, z: -0.08 };
      pts[10] = { x: wristAnchor.x - 0.01 * xSign, y: wristAnchor.y - 0.35, z: -0.05 };
      pts[11] = { x: wristAnchor.x - 0.01 * xSign, y: wristAnchor.y - 0.33, z: -0.07 };
      pts[12] = { x: wristAnchor.x - 0.01 * xSign, y: wristAnchor.y - 0.31, z: -0.08 };
      curlFinger(13, 14, 15, 16, 0.03 * xSign, 0.08);
      curlFinger(17, 18, 19, 20, 0.04 * xSign, 0.07);
      break;
    }
    case 'ily': {
      pts[3] = { x: wristAnchor.x - 0.18 * xSign, y: wristAnchor.y - 0.22, z: -0.04 };
      pts[4] = { x: wristAnchor.x - 0.25 * xSign, y: wristAnchor.y - 0.28, z: -0.05 };
      extendFinger(5, 6, 7, 8, wristAnchor.x - 0.09 * xSign, wristAnchor.y - 0.55, -0.03);
      curlFinger(9, 10, 11, 12, 0.0, 0.09);
      curlFinger(13, 14, 15, 16, 0.02 * xSign, 0.09);
      extendFinger(17, 18, 19, 20, wristAnchor.x + 0.19 * xSign, wristAnchor.y - 0.50, 0.0);
      break;
    }
    case 'stop': {
      pts[3] = { x: wristAnchor.x - 0.14 * xSign, y: wristAnchor.y - 0.18, z: -0.04 };
      pts[4] = { x: wristAnchor.x - 0.17 * xSign, y: wristAnchor.y - 0.24, z: -0.05 };
      extendFinger(5, 6, 7, 8, wristAnchor.x - 0.06 * xSign, wristAnchor.y - 0.56, -0.01);
      extendFinger(9, 10, 11, 12, wristAnchor.x - 0.01 * xSign, wristAnchor.y - 0.58, 0.0);
      extendFinger(13, 14, 15, 16, wristAnchor.x + 0.04 * xSign, wristAnchor.y - 0.57, 0.0);
      extendFinger(17, 18, 19, 20, wristAnchor.x + 0.09 * xSign, wristAnchor.y - 0.52, 0.01);
      break;
    }
    case 'ok': {
      pts[3] = { x: wristAnchor.x - 0.11 * xSign, y: wristAnchor.y - 0.26, z: -0.04 };
      pts[4] = { x: wristAnchor.x - 0.07 * xSign, y: wristAnchor.y - 0.30, z: -0.05 };
      pts[6] = { x: wristAnchor.x - 0.10 * xSign, y: wristAnchor.y - 0.34, z: -0.04 };
      pts[7] = { x: wristAnchor.x - 0.08 * xSign, y: wristAnchor.y - 0.32, z: -0.05 };
      pts[8] = { x: wristAnchor.x - 0.07 * xSign, y: wristAnchor.y - 0.30, z: -0.05 };
      extendFinger(9, 10, 11, 12, wristAnchor.x + 0.02 * xSign, wristAnchor.y - 0.56, 0.0);
      extendFinger(13, 14, 15, 16, wristAnchor.x + 0.10 * xSign, wristAnchor.y - 0.53, 0.01);
      extendFinger(17, 18, 19, 20, wristAnchor.x + 0.18 * xSign, wristAnchor.y - 0.47, 0.02);
      break;
    }
    default:
      return getSingleHandLandmarks('open', wristAnchor, isLeftHand);
  }

  return pts;
}

/**
 * Returns the raw hand landmarks array (1 or 2 hands) for a sign.
 */
function getHandsForSign(signName) {
  switch (signName) {
    case 'Hello':
      return [getSingleHandLandmarks('open', { x: 0.52, y: 0.82, z: 0.0 }, false)];

    case 'Thank You': {
      // Dual-hand gesture: both hands extending forward from chest
      const left = getSingleHandLandmarks('flat_forward', { x: 0.38, y: 0.80, z: -0.05 }, true);
      const right = getSingleHandLandmarks('flat_forward', { x: 0.62, y: 0.80, z: -0.05 }, false);
      return [left, right];
    }

    case 'Yes':
      return [getSingleHandLandmarks('fist', { x: 0.50, y: 0.78, z: -0.02 }, false)];

    case 'No':
      return [getSingleHandLandmarks('pinch_no', { x: 0.50, y: 0.76, z: -0.03 }, false)];

    case 'I Love You':
      return [getSingleHandLandmarks('ily', { x: 0.50, y: 0.82, z: 0.0 }, false)];

    case 'Stop':
      return [getSingleHandLandmarks('stop', { x: 0.50, y: 0.84, z: 0.0 }, false)];

    case 'Please': {
      // Dual-hand gesture: flat open palms facing inward near chest center
      const left = getSingleHandLandmarks('flat_forward', { x: 0.43, y: 0.78, z: 0.02 }, true);
      const right = getSingleHandLandmarks('flat_forward', { x: 0.57, y: 0.78, z: 0.02 }, false);
      return [left, right];
    }

    case 'OK':
      return [getSingleHandLandmarks('ok', { x: 0.50, y: 0.80, z: 0.0 }, false)];

    default:
      return [getSingleHandLandmarks('open', { x: 0.50, y: 0.82, z: 0.0 }, false)];
  }
}

/**
 * Generates the full 130-feature demo dataset.
 * @param {number} samplesPerClass - Default 45 samples per sign
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function generateDemoDataset(samplesPerClass = 45) {
  const signs = DEFAULT_SIGN_LABELS;
  const dataset = [];

  for (const sign of signs) {
    const rawHands = getHandsForSign(sign);
    const baseVector = normalizeLandmarks(rawHands);

    // Baseline sample
    dataset.push({
      features: baseVector,
      label: sign,
    });

    // Generate natural biometric variations
    for (let s = 1; s < samplesPerClass; s++) {
      const augmented = augmentSample(baseVector, {
        maxRotationDeg: 12,
        scaleJitter: 0.08,
        noiseStdDev: 0.016,
        allowMirror: false, // Keep canonical spatial handedness for default demo set
      });

      dataset.push({
        features: augmented,
        label: sign,
      });
    }
  }

  return dataset;
}

export const DEFAULT_SIGN_LABELS = [
  'Hello',
  'Thank You',
  'Yes',
  'No',
  'I Love You',
  'Stop',
  'Please',
  'OK',
];
