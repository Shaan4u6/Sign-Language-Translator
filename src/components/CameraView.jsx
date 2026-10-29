import React, { useEffect, useRef, useState, useCallback } from 'react';
import { getHandLandmarker, detectHandInVideo, drawHandLandmarks } from '../ml/mediapipe';
import { CameraOffIcon } from './Icons';

export function CameraView({
  onFrameLandmarks,
  isRecording = false,
  countdownText = '',
  polaroidCaption = 'Hand gesture feed',
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameId = useRef(null);

  const [cameraState, setCameraState] = useState('initializing'); // 'initializing' | 'active' | 'error'
  const [errorMessage, setErrorMessage] = useState('');
  const [handDetected, setHandDetected] = useState(false);
  const [detectedHandedness, setDetectedHandedness] = useState('');

  // Start / restart webcam stream
  const startCamera = useCallback(async () => {
    setCameraState('initializing');
    setErrorMessage('');

    try {
      getHandLandmarker().catch(err => {
        console.warn('Pre-warming HandLandmarker failed:', err);
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().catch(e => console.warn('Video play error:', e));
            setCameraState('active');
          }
        };
      }
    } catch (err) {
      console.error('Camera error:', err);
      let userMsg = 'Unable to access your webcam.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        userMsg = 'Camera permission was denied. Please allow camera access in your browser settings to use Sianor.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        userMsg = 'No camera device found. Please connect a webcam and try again.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        userMsg = 'Camera is currently in use by another application. Please close other camera apps and retry.';
      }
      setErrorMessage(userMsg);
      setCameraState('error');
    }
  }, []);

  // Initialize camera on mount
  useEffect(() => {
    let active = true;

    async function initWebcam() {
      try {
        getHandLandmarker().catch(() => {});

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (!active) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            if (active && videoRef.current) {
              videoRef.current.play().catch(() => {});
              setCameraState('active');
            }
          };
        }
      } catch (err) {
        if (!active) return;
        let userMsg = 'Unable to access your webcam.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          userMsg = 'Camera permission was denied. Please allow camera access in your browser settings to use Sianor.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          userMsg = 'No camera device found. Please connect a webcam and try again.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          userMsg = 'Camera is currently in use by another application. Please close other camera apps and retry.';
        }
        setErrorMessage(userMsg);
        setCameraState('error');
      }
    }

    initWebcam();

    return () => {
      active = false;
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  // Frame detection loop
  useEffect(() => {
    if (cameraState !== 'active') return;

    let isRunning = true;

    const processFrame = async () => {
      if (!isRunning || !videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState >= 2 && video.videoWidth > 0) {
        // Sync canvas resolution to match video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        try {
          const timestamp = performance.now();
          const detection = await detectHandInVideo(video, timestamp);

          if (detection && detection.landmarks) {
            setHandDetected(true);
            setDetectedHandedness(detection.handedness);
            drawHandLandmarks(ctx, detection.landmarks, canvas.width, canvas.height, true);

            if (onFrameLandmarks) {
              onFrameLandmarks(detection.landmarks, detection.handedness);
            }
          } else {
            setHandDetected(false);
            if (onFrameLandmarks) {
              onFrameLandmarks(null, null);
            }
          }
        } catch (procErr) {
          console.debug('Frame processing drop:', procErr);
        }
      }

      if (isRunning) {
        animFrameId.current = requestAnimationFrame(processFrame);
      }
    };

    animFrameId.current = requestAnimationFrame(processFrame);

    return () => {
      isRunning = false;
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
    };
  }, [cameraState, onFrameLandmarks]);

  return (
    <div className="polaroid-frame">
      {/* Tape decoration at the top */}
      <div className="polaroid-tape" />

      <div className="polaroid-screen-wrap">
        <video
          ref={videoRef}
          className="polaroid-video"
          playsInline
          muted
          autoPlay
        />
        <canvas
          ref={canvasRef}
          className="polaroid-canvas"
        />

        {/* Recording active overlay */}
        {isRecording && (
          <div className="recording-overlay">
            <div className="recording-indicator">
              <span className="rec-pulse-dot" />
              <span className="rec-label">RECORDING</span>
            </div>
            {countdownText && (
              <div className="recording-countdown-text">
                {countdownText}
              </div>
            )}
          </div>
        )}

        {/* Loading state */}
        {cameraState === 'initializing' && (
          <div className="camera-overlay camera-loading">
            <div className="paper-spinner" />
            <p className="loading-text">Warming up camera & hand tracker...</p>
          </div>
        )}

        {/* Error state */}
        {cameraState === 'error' && (
          <div className="camera-overlay camera-error">
            <CameraOffIcon size={40} className="error-icon" />
            <h4 className="error-title">Camera Unavailable</h4>
            <p className="error-desc">{errorMessage}</p>
            <button
              type="button"
              className="sianor-btn sianor-btn-primary"
              onClick={startCamera}
            >
              Try Again
            </button>
          </div>
        )}
      </div>

      {/* Polaroid Bottom Caption / Microcopy */}
      <div className="polaroid-caption">
        <div className="caption-text-group">
          <span className="caption-handwritten">
            {isRecording
              ? 'Keep gesture steady inside the frame...'
              : handDetected
              ? `Hand locked (${detectedHandedness})`
              : 'Show me a sign — hold hand in view'}
          </span>
          <span className="caption-sub">
            {polaroidCaption}
          </span>
        </div>

        <div className="caption-pill-status">
          <span className={`status-dot ${handDetected ? 'status-active' : 'status-waiting'}`} />
          <span className="text-xs">
            {handDetected ? 'Landmarks (21/21)' : 'Searching'}
          </span>
        </div>
      </div>
    </div>
  );
}
