/**
 * MediaPipe Tasks Vision Hand Landmarker Wrapper
 * 
 * Manages:
 * - HandLandmarker wasm runtime and neural task model initialization
 * - GPU / CPU delegation with robust local & CDN fallbacks
 * - Per-frame detection on HTML5 video elements
 * - Canvas hand skeleton and landmark drawing in warm editorial style
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

let landmarkerInstance = null;
let initPromise = null;

// Hand skeleton connection pairs (MediaPipe 21 landmark indices)
export const HAND_CONNECTIONS = [
  // Palm base
  [0, 1], [0, 5], [0, 17], [5, 9], [9, 13], [13, 17],
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index
  [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
];

export const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);

/**
 * Initializes or returns the singleton HandLandmarker instance.
 * Tries local assets first, then falls back to CDN if local fetch fails.
 * @returns {Promise<HandLandmarker>}
 */
export async function getHandLandmarker() {
  if (landmarkerInstance) return landmarkerInstance;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    // Attempt local wasm first, then fallback to jsdelivr CDN
    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    } catch (localErr) {
      console.warn('Local wasm load failed, attempting CDN fallback...', localErr);
      vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
    }

    // Attempt local model asset first, then Google Cloud Storage CDN
    const modelPaths = [
      '/models/hand_landmarker.task',
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    ];

    let landmarker = null;
    let lastError = null;

    for (const modelPath of modelPaths) {
      try {
        // Try GPU delegate first for 60fps performance
        try {
          landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: modelPath,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          break;
        } catch (gpuErr) {
          console.warn(`GPU delegate failed for ${modelPath}, trying CPU...`, gpuErr);
          landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: modelPath,
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          break;
        }
      } catch (err) {
        lastError = err;
        console.warn(`Failed loading model from ${modelPath}:`, err);
      }
    }

    if (!landmarker) {
      throw lastError || new Error('Could not load HandLandmarker model');
    }

    landmarkerInstance = landmarker;
    return landmarker;
  })();

  return initPromise;
}

/**
 * Detects hand landmarks on a video frame.
 * @param {HTMLVideoElement} video
 * @param {number} timestampMs
 * @returns {Promise<{
 *   landmarks: Array<{x: number, y: number, z: number}>,
 *   handedness: string,
 *   rawResult: any
 * } | null>}
 */
export async function detectHandInVideo(video, timestampMs) {
  if (!video || video.readyState < 2) return null;

  const landmarker = await getHandLandmarker();
  const result = landmarker.detectForVideo(video, timestampMs);

  if (result && result.landmarks && result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];
    let handedness = 'Right';
    if (result.handednesses && result.handednesses.length > 0 && result.handednesses[0].length > 0) {
      handedness = result.handednesses[0][0].categoryName || 'Right';
    }

    return {
      landmarks,
      handedness,
      rawResult: result,
    };
  }

  return null;
}

/**
 * Draws hand landmarks and skeleton connections on an HTML5 canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<{x: number, y: number, z: number}>} landmarks - MediaPipe normalized coordinates (0..1)
 * @param {number} canvasWidth
 * @param {number} canvasHeight
 * @param {boolean} mirrored - If true, flips X horizontally to match mirrored video display
 */
export function drawHandLandmarks(ctx, landmarks, canvasWidth, canvasHeight, mirrored = true) {
  if (!ctx || !landmarks || landmarks.length === 0) return;

  const getCanvasCoords = (pt) => {
    const rawX = mirrored ? (1.0 - pt.x) : pt.x;
    return {
      x: rawX * canvasWidth,
      y: pt.y * canvasHeight,
    };
  };

  ctx.save();

  // 1. Draw Skeleton Bones
  ctx.lineWidth = 3.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#C85A32'; // Terracotta accent line

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = getCanvasCoords(landmarks[startIdx]);
    const p2 = getCanvasCoords(landmarks[endIdx]);

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  // 2. Draw Subtle Inner Line for depth
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = '#FAF7F2'; // Cream inner highlight
  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = getCanvasCoords(landmarks[startIdx]);
    const p2 = getCanvasCoords(landmarks[endIdx]);

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  // 3. Draw Landmark Joints
  for (let i = 0; i < landmarks.length; i++) {
    const pos = getCanvasCoords(landmarks[i]);
    const isTip = FINGERTIP_INDICES.has(i);
    const radius = isTip ? 6.5 : (i === 0 ? 6.0 : 4.5);

    // Outer circle
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? '#E09F3E' : '#C85A32'; // Marigold for tips, Terracotta for joints
    ctx.fill();

    // Inner bright bead
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius * 0.45, 0, 2 * Math.PI);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
  }

  ctx.restore();
}
