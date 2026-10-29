/**
 * Sianor Persistence & Data Transfer Service
 * 
 * Manages:
 * - LocalStorage for custom sign labels and landmark sample dataset
 * - CSV Export: Generates downloadable comma-separated values of normalized landmark features
 * - CSV Import: Parses external gesture dataset CSV files with schema validation
 * - Model status caching
 */

const STORAGE_KEY_DATASET = 'sianor_dataset_v1';
const STORAGE_KEY_LABELS = 'sianor_labels_v1';

/**
 * Loads the dataset from localStorage.
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function loadDatasetFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_DATASET);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return parsed.map(item => ({
      label: item.label,
      features: new Float32Array(item.features),
    }));
  } catch (err) {
    console.error('Failed to load dataset from localStorage:', err);
    return [];
  }
}

/**
 * Saves the dataset to localStorage.
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 */
export function saveDatasetToStorage(dataset) {
  try {
    const serializable = dataset.map(item => ({
      label: item.label,
      features: Array.from(item.features),
    }));
    localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(serializable));
    return true;
  } catch (err) {
    console.error('Failed to save dataset to localStorage:', err);
    return false;
  }
}

/**
 * Loads configured sign labels.
 * @param {string[]} defaultLabels
 * @returns {string[]}
 */
export function loadLabelsFromStorage(defaultLabels = []) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_LABELS);
    if (!raw) return defaultLabels;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : defaultLabels;
  } catch {
    return defaultLabels;
  }
}

/**
 * Saves sign labels to storage.
 * @param {string[]} labels
 */
export function saveLabelsToStorage(labels) {
  try {
    localStorage.setItem(STORAGE_KEY_LABELS, JSON.stringify(labels));
  } catch (err) {
    console.error('Failed to save labels:', err);
  }
}

/**
 * Exports current dataset as a CSV file.
 * Format:
 * label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 * @param {string} filename
 */
export function exportDatasetToCSV(dataset, filename = 'sianor_signs_dataset.csv') {
  if (!dataset || dataset.length === 0) {
    throw new Error('Dataset is empty, nothing to export');
  }

  // Header row
  const header = ['label'];
  for (let i = 0; i < 21; i++) {
    header.push(`x${i}`, `y${i}`, `z${i}`);
  }

  const rows = [header.join(',')];

  for (const item of dataset) {
    const cleanLabel = `"${item.label.replace(/"/g, '""')}"`;
    const featureCols = Array.from(item.features).map(v => Number(v).toFixed(5));
    rows.push([cleanLabel, ...featureCols].join(','));
  }

  const csvContent = rows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Imports dataset from a CSV string.
 * @param {string} csvText
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function parseCSVToDataset(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error('CSV file has insufficient rows (requires header + data)');
  }

  const dataset = [];

  // Start from line 1 (skip header)
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Split by comma while respecting quotes
    const tokens = [];
    let current = '';
    let inQuotes = false;

    for (let c = 0; c < line.length; c++) {
      const char = line[c];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        tokens.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    tokens.push(current.trim());

    if (tokens.length < 64) {
      console.warn(`Skipping line ${i + 1}: expected 64 columns, found ${tokens.length}`);
      continue;
    }

    const rawLabel = tokens[0].replace(/^"|"$/g, '').trim();
    const features = new Float32Array(63);
    for (let f = 0; f < 63; f++) {
      const val = parseFloat(tokens[f + 1]);
      features[f] = Number.isFinite(val) ? val : 0;
    }

    dataset.push({ label: rawLabel, features });
  }

  return dataset;
}
