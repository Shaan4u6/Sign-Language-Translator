import React, { useState } from 'react';

/**
 * Confusion Matrix Heatmap for Multi-Class Evaluation.
 * Rows = Actual / True Class
 * Columns = Predicted Class
 * Shaded in warm paper & terracotta gradients.
 */
export function ConfusionMatrix({ matrix, labels }) {
  const [hoveredCell, setHoveredCell] = useState(null);

  if (!matrix || !labels || labels.length === 0) {
    return null;
  }

  // Calculate row totals (total actual samples for class r)
  const rowTotals = matrix.map(row => row.reduce((a, b) => a + b, 0));

  // Determine cell color
  const getCellBackground = (actualIdx, predIdx, count) => {
    const rowTotal = rowTotals[actualIdx] || 1;
    const ratio = count / rowTotal;

    if (count === 0) {
      return '#FAF7F2';
    }

    if (actualIdx === predIdx) {
      // Correct diagonal prediction (rich terracotta)
      const alpha = 0.15 + 0.80 * ratio;
      return `rgba(200, 90, 50, ${alpha.toFixed(2)})`;
    } else {
      // Off-diagonal error (amber warning shade)
      const alpha = 0.20 + 0.65 * ratio;
      return `rgba(224, 159, 62, ${alpha.toFixed(2)})`;
    }
  };

  const getCellTextColor = (actualIdx, predIdx, count) => {
    const rowTotal = rowTotals[actualIdx] || 1;
    const ratio = count / rowTotal;
    if (actualIdx === predIdx && ratio > 0.5) {
      return '#FFFFFF';
    }
    return '#1F1B18';
  };

  return (
    <div className="confusion-matrix-wrapper">
      <div className="cm-header-meta">
        <span className="cm-axis-title">Actual vs. Predicted Heatmap</span>
        <span className="cm-legend-hint">Diagonal = Correct classifications</span>
      </div>

      <div className="cm-scroll-area">
        <table className="cm-table">
          <thead>
            <tr>
              <th className="cm-corner-cell">
                <span className="cm-axis-actual">Actual ↓</span>
                <span className="cm-axis-pred">Pred →</span>
              </th>
              {labels.map(lbl => (
                <th key={`col-${lbl}`} className="cm-col-header" title={`Predicted: ${lbl}`}>
                  <span className="cm-header-text">{lbl}</span>
                </th>
              ))}
              <th className="cm-total-header">Total</th>
            </tr>
          </thead>
          <tbody>
            {labels.map((rowLabel, rIdx) => {
              const totalActual = rowTotals[rIdx];
              return (
                <tr key={`row-${rowLabel}`}>
                  <th className="cm-row-header" title={`Actual: ${rowLabel}`}>
                    {rowLabel}
                  </th>
                  {labels.map((colLabel, cIdx) => {
                    const count = matrix[rIdx][cIdx] || 0;
                    const pct = totalActual > 0 ? Math.round((count / totalActual) * 100) : 0;
                    const isDiagonal = rIdx === cIdx;
                    const isHovered = hoveredCell && hoveredCell.r === rIdx && hoveredCell.c === cIdx;

                    return (
                      <td
                        key={`cell-${rIdx}-${cIdx}`}
                        className={`cm-cell ${isDiagonal ? 'cm-cell-diagonal' : ''} ${isHovered ? 'cm-cell-hover' : ''}`}
                        style={{
                          backgroundColor: getCellBackground(rIdx, cIdx, count),
                          color: getCellTextColor(rIdx, cIdx, count),
                        }}
                        onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, count, pct, actual: rowLabel, pred: colLabel })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        <span className="cm-cell-count">{count}</span>
                        {totalActual > 0 && count > 0 && (
                          <span className="cm-cell-pct">{pct}%</span>
                        )}
                      </td>
                    );
                  })}
                  <td className="cm-row-total-cell">
                    <strong>{totalActual}</strong>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {hoveredCell && (
        <div className="cm-tooltip-card">
          <span>
            Actual: <strong>{hoveredCell.actual}</strong> &nbsp;→&nbsp; Predicted: <strong>{hoveredCell.pred}</strong>
          </span>
          <span className="cm-tooltip-metrics">
            Count: <strong>{hoveredCell.count}</strong> ({hoveredCell.pct}% of class)
          </span>
        </div>
      )}
    </div>
  );
}
