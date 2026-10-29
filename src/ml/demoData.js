/**
 * Sianor Pre-recorded / Synthesized Realistic Demo Dataset
 * 
 * Provides calibrated 21-landmark poses for all 8 default signs:
 * 1. Hello: Open hand, fingers spread upward
 * 2. Thank You: Flat hand angled forward
 * 3. Yes: Closed fist (nodding pose)
 * 4. No: Index and middle finger pinch towards thumb
 * 5. I Love You: Thumb, index, and pinky extended (ASL ILY)
 * 6. Stop: Open palm vertical, fingers together, facing camera
 * 7. Please: Flat open palm gently curved
 * 8. OK: Thumb and index fingertips touching in an "O", remaining 3 fingers straight
 * 
 * Generates ~45 high-fidelity samples per class (~360 total) with realistic anatomical
 * variations, allowing instant training and evaluation without requiring manual recording.
 */

import { normalizeLandmarks, augmentSample } from './preprocessor';

/**
 * Base anatomical landmark templates (21 points [x, y, z]) for a right hand.
 * Coordinate frame: x increases right, y increases downward, z increases toward camera.
 */
function getBaseLandmarks(signName) {
  // Base wrist point at center-bottom
  const wrist = { x: 0.5, y: 0.85, z: 0.0 };

  // Common MCP (knuckle) anchor positions relative to wrist
  const thumbCmc = { x: 0.42, y: 0.77, z: -0.02 };
  const thumbMcp = { x: 0.36, y: 0.69, z: -0.04 };
  const indexMcp = { x: 0.44, y: 0.58, z: -0.02 };
  const middleMcp = { x: 0.50, y: 0.56, z: -0.01 };
  const ringMcp = { x: 0.56, y: 0.58, z: 0.0 };
  const pinkyMcp = { x: 0.62, y: 0.62, z: 0.01 };

  const pts = new Array(21);
  pts[0] = wrist;
  pts[1] = thumbCmc;
  pts[2] = thumbMcp;
  pts[5] = indexMcp;
  pts[9] = middleMcp;
  pts[13] = ringMcp;
  pts[17] = pinkyMcp;

  // Helper to extend a finger upward
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

  // Helper to curl a finger down into a fist
  const curlFinger = (mcpIdx, pipIdx, dipIdx, tipIdx, inwardX = 0, inwardY = 0.08) => {
    const mcp = pts[mcpIdx];
    pts[pipIdx] = { x: mcp.x + inwardX * 0.5, y: mcp.y - 0.06, z: -0.08 };
    pts[dipIdx] = { x: mcp.x + inwardX, y: mcp.y + inwardY * 0.6, z: -0.05 };
    pts[tipIdx] = { x: mcp.x + inwardX * 0.8, y: mcp.y + inwardY, z: -0.02 };
  };

  switch (signName) {
    case 'Hello': {
      // Open hand, all fingers extended upward and slightly spread
      pts[3] = { x: 0.30, y: 0.62, z: -0.05 };
      pts[4] = { x: 0.24, y: 0.56, z: -0.06 }; // Thumb extended out
      extendFinger(5, 6, 7, 8, 0.38, 0.30, -0.03);    // Index
      extendFinger(9, 10, 11, 12, 0.50, 0.26, -0.02);  // Middle
      extendFinger(13, 14, 15, 16, 0.61, 0.30, -0.01); // Ring
      extendFinger(17, 18, 19, 20, 0.72, 0.37, 0.0);   // Pinky
      break;
    }

    case 'Thank You': {
      // Flat hand tilted slightly forward from chin/chest
      pts[3] = { x: 0.36, y: 0.62, z: -0.06 };
      pts[4] = { x: 0.32, y: 0.56, z: -0.08 }; // Thumb tucked slightly alongside
      extendFinger(5, 6, 7, 8, 0.44, 0.36, -0.15);    // Fingers reaching forward (z negative)
      extendFinger(9, 10, 11, 12, 0.50, 0.34, -0.15);
      extendFinger(13, 14, 15, 16, 0.56, 0.37, -0.14);
      extendFinger(17, 18, 19, 20, 0.63, 0.42, -0.12);
      break;
    }

    case 'Yes': {
      // Closed fist, knuckles facing camera, thumb curled over index
      pts[3] = { x: 0.40, y: 0.63, z: -0.08 };
      pts[4] = { x: 0.45, y: 0.60, z: -0.10 }; // Thumb resting on curled fingers
      curlFinger(5, 6, 7, 8, -0.02, 0.08);
      curlFinger(9, 10, 11, 12, 0.0, 0.08);
      curlFinger(13, 14, 15, 16, 0.02, 0.08);
      curlFinger(17, 18, 19, 20, 0.03, 0.07);
      break;
    }

    case 'No': {
      // Index and middle fingers touch thumb tip, ring and pinky curled
      pts[3] = { x: 0.44, y: 0.55, z: -0.07 };
      pts[4] = { x: 0.48, y: 0.52, z: -0.08 }; // Thumb tip
      // Index touches thumb
      pts[6] = { x: 0.43, y: 0.50, z: -0.06 };
      pts[7] = { x: 0.46, y: 0.51, z: -0.07 };
      pts[8] = { x: 0.48, y: 0.52, z: -0.08 };
      // Middle touches thumb
      pts[10] = { x: 0.49, y: 0.49, z: -0.05 };
      pts[11] = { x: 0.49, y: 0.51, z: -0.07 };
      pts[12] = { x: 0.49, y: 0.53, z: -0.08 };
      // Ring and pinky curled
      curlFinger(13, 14, 15, 16, 0.03, 0.08);
      curlFinger(17, 18, 19, 20, 0.04, 0.07);
      break;
    }

    case 'I Love You': {
      // ASL ILY: Thumb, Index, Pinky extended, Middle and Ring curled
      pts[3] = { x: 0.32, y: 0.62, z: -0.04 };
      pts[4] = { x: 0.25, y: 0.56, z: -0.05 }; // Thumb out
      extendFinger(5, 6, 7, 8, 0.41, 0.29, -0.03);    // Index straight UP
      curlFinger(9, 10, 11, 12, 0.0, 0.09);            // Middle curled DOWN
      curlFinger(13, 14, 15, 16, 0.02, 0.09);          // Ring curled DOWN
      extendFinger(17, 18, 19, 20, 0.69, 0.34, 0.0);   // Pinky straight UP
      break;
    }

    case 'Stop': {
      // Open vertical palm, fingers held tightly together pointing straight up
      pts[3] = { x: 0.36, y: 0.66, z: -0.04 };
      pts[4] = { x: 0.33, y: 0.60, z: -0.05 }; // Thumb slightly alongside palm
      extendFinger(5, 6, 7, 8, 0.44, 0.28, -0.01);    // Index
      extendFinger(9, 10, 11, 12, 0.49, 0.26, 0.0);    // Middle
      extendFinger(13, 14, 15, 16, 0.54, 0.27, 0.0);   // Ring
      extendFinger(17, 18, 19, 20, 0.59, 0.32, 0.01);  // Pinky (close together)
      break;
    }

    case 'Please': {
      // Flat hand held over chest in circular/gentle diagonal orientation
      pts[3] = { x: 0.38, y: 0.67, z: -0.03 };
      pts[4] = { x: 0.32, y: 0.63, z: -0.04 };
      extendFinger(5, 6, 7, 8, 0.42, 0.38, 0.03);
      extendFinger(9, 10, 11, 12, 0.48, 0.36, 0.04);
      extendFinger(13, 14, 15, 16, 0.54, 0.38, 0.04);
      extendFinger(17, 18, 19, 20, 0.60, 0.42, 0.04);
      break;
    }

    case 'OK': {
      // Index and Thumb form an "O", Middle, Ring, Pinky upright and spread
      // Thumb tip touches index tip
      pts[3] = { x: 0.39, y: 0.58, z: -0.04 };
      pts[4] = { x: 0.43, y: 0.54, z: -0.05 }; // Thumb tip
      pts[6] = { x: 0.40, y: 0.50, z: -0.04 };
      pts[7] = { x: 0.42, y: 0.52, z: -0.05 };
      pts[8] = { x: 0.43, y: 0.54, z: -0.05 }; // Index tip touches thumb tip!
      // Other 3 fingers spread upright
      extendFinger(9, 10, 11, 12, 0.52, 0.28, 0.0);
      extendFinger(13, 14, 15, 16, 0.60, 0.31, 0.01);
      extendFinger(17, 18, 19, 20, 0.68, 0.37, 0.02);
      break;
    }

    default: {
      return getBaseLandmarks('Hello');
    }
  }

  return pts;
}

/**
 * Generates the full demo dataset with diverse variations per sign.
 * @param {number} samplesPerClass - Default 45 samples per sign
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function generateDemoDataset(samplesPerClass = 45) {
  const signs = ['Hello', 'Thank You', 'Yes', 'No', 'I Love You', 'Stop', 'Please', 'OK'];
  const dataset = [];

  for (const sign of signs) {
    const basePts = getBaseLandmarks(sign);
    // Baseline normalized vector
    const baseVector = normalizeLandmarks(basePts, 'Right');

    // Add base sample
    dataset.push({
      features: baseVector,
      label: sign,
    });

    // Generate natural biometric variations (subtle hand movements, rotation, scale, noise)
    for (let s = 1; s < samplesPerClass; s++) {
      const augmented = augmentSample(baseVector, {
        maxRotationDeg: 14,
        scaleJitter: 0.10,
        noiseStdDev: 0.018,
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
