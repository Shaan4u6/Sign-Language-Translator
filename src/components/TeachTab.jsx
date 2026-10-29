import React, { useState, useRef, useEffect, useCallback } from 'react';
import { CameraView } from './CameraView';
import { normalizeLandmarks } from '../ml/preprocessor';
import { generateDemoDataset } from '../ml/demoData';
import {
  exportDatasetToCSV,
  parseCSVToDataset,
  saveDatasetToStorage,
  saveLabelsToStorage,
} from '../ml/storage';
import {
  RecordCircleIcon,
  TrashIcon,
  PlusIcon,
  EditIcon,
  DownloadIcon,
  UploadIcon,
  SparklesIcon,
  CheckIcon,
} from './Icons';
import confetti from 'canvas-confetti';

export function TeachTab({
  labels,
  setLabels,
  dataset,
  setDataset,
}) {
  // Label management state
  const [newLabelInput, setNewLabelInput] = useState('');
  const [editingLabel, setEditingLabel] = useState(null); // original label string
  const [editInput, setEditInput] = useState('');

  // Recording lifecycle state
  const [recordingLabel, setRecordingLabel] = useState(null);
  const [countdownPhase, setCountdownPhase] = useState('idle'); // 'prep' (3s) | 'recording' (10s) | 'idle'
  const [countdownSeconds, setCountdownSeconds] = useState(0);
  const [recordedThisSession, setRecordedThisSession] = useState(0);

  const accumulatedSamplesRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Calculate sample counts per label
  const sampleCounts = React.useMemo(() => {
    const counts = {};
    for (const l of labels) counts[l] = 0;
    for (const item of dataset) {
      counts[item.label] = (counts[item.label] || 0) + 1;
    }
    return counts;
  }, [labels, dataset]);

  // Frame listener for landmark recording
  const handleFrameLandmarks = useCallback((landmarks) => {
    if (countdownPhase !== 'recording' || !recordingLabel || !landmarks) return;

    try {
      const vector = normalizeLandmarks(landmarks);
      if (!vector) return;
      accumulatedSamplesRef.current.push({
        label: recordingLabel,
        features: vector,
      });
      setRecordedThisSession(accumulatedSamplesRef.current.length);
    } catch (e) {
      console.warn('Frame normalization drop:', e);
    }
  }, [countdownPhase, recordingLabel]);

  // Start recording workflow for a given label
  const startRecordingWorkflow = (label) => {
    if (countdownPhase !== 'idle') return;

    setRecordingLabel(label);
    setCountdownPhase('prep');
    setCountdownSeconds(3);
    accumulatedSamplesRef.current = [];
    setRecordedThisSession(0);

    let prepCount = 3;
    const prepInterval = setInterval(() => {
      prepCount--;
      if (prepCount > 0) {
        setCountdownSeconds(prepCount);
      } else {
        clearInterval(prepInterval);
        // Begin active 10s recording
        setCountdownPhase('recording');
        setCountdownSeconds(10);

        let recCount = 10;
        const recInterval = setInterval(() => {
          recCount--;
          if (recCount > 0) {
            setCountdownSeconds(recCount);
          } else {
            clearInterval(recInterval);
            finishRecording(label);
          }
        }, 1000);
        recordingTimerRef.current = recInterval;
      }
    }, 1000);
    recordingTimerRef.current = prepInterval;
  };

  // Finish and commit recorded samples
  const finishRecording = () => {
    const newSamples = accumulatedSamplesRef.current;
    if (newSamples.length > 0) {
      setDataset(prev => {
        const updated = [...prev, ...newSamples];
        saveDatasetToStorage(updated);
        return updated;
      });

      try {
        confetti({
          particleCount: 40,
          spread: 60,
          origin: { y: 0.8 },
          colors: ['#C85A32', '#E09F3E', '#FAF7F2'],
        });
      } catch {
        // Confetti optional
      }
    }

    setCountdownPhase('idle');
    setRecordingLabel(null);
    setCountdownSeconds(0);
  };

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  // Add new label
  const handleAddLabel = (e) => {
    e.preventDefault();
    const trimmed = newLabelInput.trim();
    if (!trimmed) return;
    if (labels.some(l => l.toLowerCase() === trimmed.toLowerCase())) {
      alert(`Label "${trimmed}" already exists.`);
      return;
    }
    const updated = [...labels, trimmed];
    setLabels(updated);
    saveLabelsToStorage(updated);
    setNewLabelInput('');
  };

  // Delete label and its samples
  const handleDeleteLabel = (labelToDelete) => {
    const count = sampleCounts[labelToDelete] || 0;
    const confirmMsg = count > 0
      ? `Delete sign "${labelToDelete}" and its ${count} recorded samples?`
      : `Delete sign "${labelToDelete}"?`;
    if (!window.confirm(confirmMsg)) return;

    const updatedLabels = labels.filter(l => l !== labelToDelete);
    const updatedDataset = dataset.filter(item => item.label !== labelToDelete);
    setLabels(updatedLabels);
    setDataset(updatedDataset);
    saveLabelsToStorage(updatedLabels);
    saveDatasetToStorage(updatedDataset);
  };

  // Rename label
  const handleStartRename = (lbl) => {
    setEditingLabel(lbl);
    setEditInput(lbl);
  };

  const handleSaveRename = (e) => {
    e.preventDefault();
    const trimmed = editInput.trim();
    if (!trimmed || trimmed === editingLabel) {
      setEditingLabel(null);
      return;
    }
    if (labels.some(l => l.toLowerCase() === trimmed.toLowerCase() && l !== editingLabel)) {
      alert(`Label "${trimmed}" already exists.`);
      return;
    }

    const updatedLabels = labels.map(l => (l === editingLabel ? trimmed : l));
    const updatedDataset = dataset.map(item => (item.label === editingLabel ? { ...item, label: trimmed } : item));

    setLabels(updatedLabels);
    setDataset(updatedDataset);
    saveLabelsToStorage(updatedLabels);
    saveDatasetToStorage(updatedDataset);
    setEditingLabel(null);
  };

  // Load demo dataset
  const handleLoadDemoDataset = () => {
    const existingCount = dataset.length;
    if (existingCount > 0) {
      if (!window.confirm(`Load demo dataset? This will append ~360 pre-calibrated samples for the 8 standard signs.`)) {
        return;
      }
    }

    const demo = generateDemoDataset(45); // 45 samples * 8 signs = 360 samples
    const updated = [...dataset, ...demo];
    setDataset(updated);
    saveDatasetToStorage(updated);

    confetti({
      particleCount: 75,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#C85A32', '#E09F3E', '#4E6E58'],
    });
  };

  // CSV Export
  const handleExportCSV = () => {
    if (dataset.length === 0) {
      alert('Dataset is currently empty. Record samples or load the demo dataset first.');
      return;
    }
    exportDatasetToCSV(dataset);
  };

  // CSV Import
  const handleImportFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const parsed = parseCSVToDataset(text);
        if (parsed.length === 0) {
          alert('No valid landmark samples found in CSV.');
          return;
        }

        // Merge imported labels into labels list
        const importedLabels = Array.from(new Set(parsed.map(p => p.label)));
        const combinedLabels = Array.from(new Set([...labels, ...importedLabels]));

        const combinedDataset = [...dataset, ...parsed];
        setLabels(combinedLabels);
        setDataset(combinedDataset);
        saveLabelsToStorage(combinedLabels);
        saveDatasetToStorage(combinedDataset);

        alert(`Successfully imported ${parsed.length} samples across ${importedLabels.length} sign labels!`);
      } catch (err) {
        alert(`Failed to import CSV: ${err.message}`);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Clear all samples
  const handleClearAllData = () => {
    if (dataset.length === 0) return;
    if (!window.confirm(`Are you sure you want to delete ALL ${dataset.length} samples in your dataset? This cannot be undone.`)) {
      return;
    }
    setDataset([]);
    saveDatasetToStorage([]);
  };

  // Countdown badge overlay text for polaroid
  let countdownText = '';
  if (countdownPhase === 'prep') {
    countdownText = `Get ready... ${countdownSeconds}`;
  } else if (countdownPhase === 'recording') {
    countdownText = `Recording "${recordingLabel}": ${countdownSeconds}s (${recordedThisSession} frames)`;
  }

  return (
    <div className="teach-layout">
      {/* Left Column: Live Webcam with Recording Feedback */}
      <section className="camera-column" aria-label="Webcam Recording Feed">
        <CameraView
          onFrameLandmarks={handleFrameLandmarks}
          isRecording={countdownPhase === 'recording'}
          countdownText={countdownText}
          polaroidCaption={
            countdownPhase === 'recording'
              ? `Capturing 21 landmarks for "${recordingLabel}"`
              : 'Stand 2-3 feet away, hand centered'
          }
        />

        {/* Dataset Action Utilities */}
        <div className="teach-quick-actions">
          <button
            type="button"
            className="sianor-btn sianor-btn-demo"
            onClick={handleLoadDemoDataset}
            title="Load calibrated baseline samples for the 8 default signs"
          >
            <SparklesIcon size={17} />
            <span>Load Demo Dataset (360 samples)</span>
          </button>

          <div className="import-export-row">
            <button
              type="button"
              className="sianor-btn sianor-btn-secondary btn-sm"
              onClick={handleExportCSV}
              disabled={dataset.length === 0}
            >
              <DownloadIcon size={15} />
              <span>Export CSV</span>
            </button>

            <button
              type="button"
              className="sianor-btn sianor-btn-secondary btn-sm"
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <UploadIcon size={15} />
              <span>Import CSV</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleImportFileChange}
            />

            {dataset.length > 0 && (
              <button
                type="button"
                className="sianor-btn sianor-btn-ghost btn-sm"
                onClick={handleClearAllData}
                title="Clear all recorded data"
              >
                Clear All
              </button>
            )}
          </div>
        </div>
      </section>

      {/* Right Column: Sign Labels Management & Record Buttons */}
      <section className="output-column" aria-label="Sign Labels List">
        <div className="sianor-card labels-card">
          <div className="card-header-row">
            <div>
              <span className="card-kicker">Vocabulary Collection</span>
              <h3 className="card-title-sm">Sign Labels & Recording</h3>
            </div>
            <div className="dataset-total-badge">
              <span className="total-num">{dataset.length}</span>
              <span className="total-sub">total samples</span>
            </div>
          </div>

          <p className="card-instruction">
            Click <strong>Record</strong> next to any sign. You'll get a 3-second countdown,
            then 10 seconds of continuous landmark sampling (~200 frames). Keep your hand steady!
          </p>

          {/* Add New Label Form */}
          <form className="add-label-form" onSubmit={handleAddLabel}>
            <input
              type="text"
              placeholder="Add custom sign (e.g. 'Help', 'Water')..."
              value={newLabelInput}
              onChange={e => setNewLabelInput(e.target.value)}
              className="sianor-input"
              maxLength={24}
            />
            <button
              type="submit"
              className="sianor-btn sianor-btn-primary"
              disabled={!newLabelInput.trim()}
            >
              <PlusIcon size={16} />
              <span>Add Sign</span>
            </button>
          </form>

          {/* Labels List */}
          <div className="labels-list">
            {labels.map(lbl => {
              const count = sampleCounts[lbl] || 0;
              const isCurrentlyRecording = recordingLabel === lbl && countdownPhase !== 'idle';
              const isEditing = editingLabel === lbl;

              return (
                <div
                  key={lbl}
                  className={`label-item ${isCurrentlyRecording ? 'label-item-recording' : ''}`}
                >
                  <div className="label-item-main">
                    {isEditing ? (
                      <form className="rename-form" onSubmit={handleSaveRename}>
                        <input
                          type="text"
                          value={editInput}
                          onChange={e => setEditInput(e.target.value)}
                          className="sianor-input rename-input"
                          autoFocus
                        />
                        <button type="submit" className="sianor-btn sianor-btn-primary btn-sm">
                          <CheckIcon size={14} />
                        </button>
                        <button
                          type="button"
                          className="sianor-btn sianor-btn-ghost btn-sm"
                          onClick={() => setEditingLabel(null)}
                        >
                          Cancel
                        </button>
                      </form>
                    ) : (
                      <div className="label-title-group">
                        <span className="label-name">{lbl}</span>
                        <span className={`label-count-tag ${count >= 30 ? 'tag-good' : count > 0 ? 'tag-low' : 'tag-empty'}`}>
                          {count} {count === 1 ? 'sample' : 'samples'}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Actions for this label */}
                  <div className="label-item-actions">
                    <button
                      type="button"
                      className={`sianor-btn ${isCurrentlyRecording ? 'sianor-btn-rec-active' : 'sianor-btn-record'}`}
                      onClick={() => startRecordingWorkflow(lbl)}
                      disabled={countdownPhase !== 'idle'}
                      title={`Record 10 seconds of samples for "${lbl}"`}
                    >
                      <RecordCircleIcon size={16} />
                      <span>{isCurrentlyRecording ? `${countdownSeconds}s...` : 'Record'}</span>
                    </button>

                    {!isEditing && (
                      <>
                        <button
                          type="button"
                          className="icon-btn"
                          onClick={() => handleStartRename(lbl)}
                          disabled={countdownPhase !== 'idle'}
                          title="Rename this sign"
                        >
                          <EditIcon size={15} />
                        </button>

                        <button
                          type="button"
                          className="icon-btn icon-btn-danger"
                          onClick={() => handleDeleteLabel(lbl)}
                          disabled={countdownPhase !== 'idle'}
                          title="Delete this sign and samples"
                        >
                          <TrashIcon size={15} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="teach-tip-footer">
            <span className="tip-bullet">•</span>
            <span>
              Tip: Aim for at least <strong>30–50 samples</strong> per sign with slight natural hand tilts
              so the model generalizes well.
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
