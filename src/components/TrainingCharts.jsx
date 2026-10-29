import React, { useEffect, useRef } from 'react';

/**
 * Live HTML5 Canvas Chart for Training Loss & Accuracy per Epoch.
 * Designed with Sianor's warm editorial paper styling.
 */
export function TrainingCharts({ epochHistory, maxEpochs = 50 }) {
  const lossCanvasRef = useRef(null);
  const accCanvasRef = useRef(null);

  // Render Loss Chart
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas || epochHistory.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 20, right: 25, bottom: 30, left: 45 };

    ctx.clearRect(0, 0, width, height);

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Determine max loss
    const allLosses = epochHistory.flatMap(h => [h.loss, h.valLoss].filter(Number.isFinite));
    const maxVal = Math.max(1.5, ...allLosses);

    // Draw horizontal grid lines
    ctx.strokeStyle = '#EBE4D5';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#7C7267';
    ctx.font = '11px "DM Sans", sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartH / 4) * i;
      const val = (maxVal * (1 - i / 4)).toFixed(2);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(val, padding.left - 6, y + 4);
    }

    // X-axis baseline
    ctx.beginPath();
    ctx.strokeStyle = '#D5CCBC';
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw curves
    const totalPoints = Math.max(maxEpochs, epochHistory.length);

    const drawCurve = (prop, strokeStyle) => {
      ctx.beginPath();
      epochHistory.forEach((item, idx) => {
        const val = item[prop];
        if (!Number.isFinite(val)) return;
        const x = padding.left + (idx / (totalPoints - 1 || 1)) * chartW;
        const y = padding.top + (1 - Math.min(val, maxVal) / maxVal) * chartH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    };

    // Train loss (Terracotta)
    drawCurve('loss', '#C85A32');
    // Val loss (Marigold)
    drawCurve('valLoss', '#E09F3E');
  }, [epochHistory, maxEpochs]);

  // Render Accuracy Chart
  useEffect(() => {
    const canvas = accCanvasRef.current;
    if (!canvas || epochHistory.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 20, right: 25, bottom: 30, left: 45 };

    ctx.clearRect(0, 0, width, height);

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Draw horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    ctx.strokeStyle = '#EBE4D5';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#7C7267';
    ctx.font = '11px "DM Sans", sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartH / 4) * i;
      const pct = (100 * (1 - i / 4)).toFixed(0) + '%';
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(pct, padding.left - 6, y + 4);
    }

    // X-axis baseline
    ctx.beginPath();
    ctx.strokeStyle = '#D5CCBC';
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    const totalPoints = Math.max(maxEpochs, epochHistory.length);

    const drawCurve = (prop, strokeStyle) => {
      ctx.beginPath();
      epochHistory.forEach((item, idx) => {
        const val = item[prop];
        if (!Number.isFinite(val)) return;
        const x = padding.left + (idx / (totalPoints - 1 || 1)) * chartW;
        const y = padding.top + (1 - Math.min(val, 1.0)) * chartH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    };

    // Train Acc (Terracotta)
    drawCurve('accuracy', '#C85A32');
    // Val Acc (Marigold)
    drawCurve('valAccuracy', '#E09F3E');
  }, [epochHistory, maxEpochs]);

  return (
    <div className="charts-container">
      <div className="chart-card">
        <div className="chart-header">
          <span className="chart-title">Loss per Epoch (Categorical Crossentropy)</span>
          <div className="chart-legend">
            <span className="legend-item"><span className="legend-dot dot-train" />Train Loss</span>
            <span className="legend-item"><span className="legend-dot dot-val" />Validation</span>
          </div>
        </div>
        <canvas
          ref={lossCanvasRef}
          width={420}
          height={180}
          className="metric-canvas"
        />
      </div>

      <div className="chart-card">
        <div className="chart-header">
          <span className="chart-title">Accuracy per Epoch</span>
          <div className="chart-legend">
            <span className="legend-item"><span className="legend-dot dot-train" />Train Acc</span>
            <span className="legend-item"><span className="legend-dot dot-val" />Validation</span>
          </div>
        </div>
        <canvas
          ref={accCanvasRef}
          width={420}
          height={180}
          className="metric-canvas"
        />
      </div>
    </div>
  );
}
