/**
 * Sianor Persistence & Data Transfer Service (130-Feature Dual-Hand Version)
 * 
 * Manages:
 * - LocalStorage for custom sign labels and landmark sample dataset
 * - Legacy data detection and automatic purge if feature dimension !== 130
 * - CSV Export: Generates downloadable comma-separated values of 130 dual-hand features
 * - CSV Import: Parses dual-hand gesture dataset CSV files (130 features + label)
 * - Auto-loading default dataset from /dataset.csv on initial startup
 */

const STORAGE_KEY_DATASET = 'sianor_dataset_v2_dual';
const STORAGE_KEY_LABELS = 'sianor_labels_v2_dual';
const LEGACY_STORAGE_KEY_DATASET = 'sianor_dataset_v1';
const LEGACY_STORAGE_KEY_LABELS = 'sianor_labels_v1';

export const FEATURE_SIZE = 130;

/**
 * Loads the dataset from localStorage.
 * Automatically clears legacy 63-feature datasets.
 * @returns {{ dataset: Array<{features: Float32Array, label: string}>, legacyCleared: boolean }}
 */
export function loadDatasetFromStorage() {
  let legacyCleared = false;

  // Check for legacy v1 storage and purge it
  if (localStorage.getItem(LEGACY_STORAGE_KEY_DATASET)) {
    localStorage.removeItem(LEGACY_STORAGE_KEY_DATASET);
    localStorage.removeItem(LEGACY_STORAGE_KEY_LABELS);
    legacyCleared = true;
  }

  try {
    const raw = localStorage.getItem(STORAGE_KEY_DATASET);
    if (!raw) return { dataset: [], legacyCleared };
    const parsed = JSON.parse(raw);

    // Verify all items have exactly 130 features
    const isValid = parsed.every(item => item.features && item.features.length === FEATURE_SIZE);
    if (!isValid) {
      console.warn(`Stored dataset contains non-130 feature samples. Purging legacy dataset...`);
      localStorage.removeItem(STORAGE_KEY_DATASET);
      return { dataset: [], legacyCleared: true };
    }

    const dataset = parsed.map(item => ({
      label: item.label,
      features: new Float32Array(item.features),
    }));

    return { dataset, legacyCleared };
  } catch (err) {
    console.error('Failed to load dataset from localStorage:', err);
    return { dataset: [], legacyCleared };
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
 * Exports current dual-hand dataset as a CSV file.
 * Format:
 * label,h1_x0,h1_y0,h1_z0...h1_z20,h2_x0,h2_y0...h2_z20,offset_dx,offset_dy,offset_dz,two_hands_flag (131 columns)
 * @param {Array<{features: Float32Array|number[], label: string}>} dataset
 * @param {string} filename
 */
export function exportDatasetToCSV(dataset, filename = 'sianor_dual_hand_dataset.csv') {
  if (!dataset || dataset.length === 0) {
    throw new Error('Dataset is empty, nothing to export');
  }

  // Header row (130 features)
  const header = ['label'];

  // Hand 1 (Slot 1)
  for (let i = 0; i < 21; i++) {
    header.push(`h1_x${i}`, `h1_y${i}`, `h1_z${i}`);
  }

  // Hand 2 (Slot 2)
  for (let i = 0; i < 21; i++) {
    header.push(`h2_x${i}`, `h2_y${i}`, `h2_z${i}`);
  }

  // 4 Extra Features
  header.push('offset_dx', 'offset_dy', 'offset_dz', 'two_hands_flag');

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
 * Imports dual-hand dataset from a CSV string.
 * Validates 130 feature columns + label = 131 columns.
 * @param {string} csvText
 * @returns {Array<{features: Float32Array, label: string}>}
 */
export function parseCSVToDataset(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error('CSV file has insufficient rows (requires header + data)');
  }

  const dataset = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Split by comma respecting quotes
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

    if (tokens.length < (FEATURE_SIZE + 1)) {
      console.warn(`Skipping line ${i + 1}: expected ${FEATURE_SIZE + 1} columns, found ${tokens.length}`);
      continue;
    }

    const rawLabel = tokens[0].replace(/^"|"$/g, '').trim();
    const features = new Float32Array(FEATURE_SIZE);
    for (let f = 0; f < FEATURE_SIZE; f++) {
      const val = parseFloat(tokens[f + 1]);
      features[f] = Number.isFinite(val) ? val : 0;
    }

    dataset.push({ label: rawLabel, features });
  }

  return dataset;
}

/**
 * Checks if /dataset.csv exists in public root and loads it if found and matching 130 features.
 * @returns {Promise<{ dataset: Array<{features: Float32Array, label: string}>, labels: string[] } | null>}
 */
export async function checkAndLoadDefaultDatasetCSV() {
  try {
    const res = await fetch('/dataset.csv');
    if (!res.ok) return null;
    const text = await res.text();
    const dataset = parseCSVToDataset(text);
    if (!dataset || dataset.length === 0) return null;

    const labels = Array.from(new Set(dataset.map(d => d.label)));
    return { dataset, labels };
  } catch (err) {
    console.debug('No default /dataset.csv found or failed to parse:', err);
    return null;
  }
}
