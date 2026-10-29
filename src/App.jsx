import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { TranslateTab } from './components/TranslateTab';
import { TeachTab } from './components/TeachTab';
import { TrainTab } from './components/TrainTab';
import { AboutTab } from './components/AboutTab';

import { DEFAULT_SIGN_LABELS } from './ml/demoData';
import {
  loadDatasetFromStorage,
  saveDatasetToStorage,
  loadLabelsFromStorage,
  saveLabelsToStorage,
  checkAndLoadDefaultDatasetCSV,
} from './ml/storage';
import { KNNClassifier } from './ml/knn';
import { TFHandModel } from './ml/tfModel';
import { SparklesIcon } from './components/Icons';
import './styles/sianor.css';

export function App() {
  const [activeTab, setActiveTab] = useState('translate');

  // Signs vocabulary
  const [labels, setLabels] = useState(() => loadLabelsFromStorage(DEFAULT_SIGN_LABELS));

  // Dataset of landmark samples: Array<{ label: string, features: Float32Array }>
  const [dataset, setDataset] = useState(() => {
    const { dataset } = loadDatasetFromStorage();
    return dataset;
  });

  // Upgrade / legacy clearing notice banner
  const [upgradeNotice, setUpgradeNotice] = useState(() => {
    const { legacyCleared } = loadDatasetFromStorage();
    return legacyCleared
      ? 'Upgraded to Sianor Dual-Hand (130 features). Legacy 63-feature data was cleared to ensure smooth operation.'
      : null;
  });

  // Model instances
  const [knnModel, setKnnModel] = useState(() => new KNNClassifier(5));
  const [tfModel, setTfModel] = useState(() => new TFHandModel());

  // Model selection & readiness
  const [activeModelType, setActiveModelType] = useState('mlp'); // 'mlp' | 'knn'
  const [modelStatus, setModelStatus] = useState({
    trained: false,
    modelType: 'mlp',
    labels: [],
    accuracy: null,
  });

  // On mount: check for /dataset.csv and restore saved model from IndexedDB
  useEffect(() => {
    let isMounted = true;

    async function initializeDataAndModel() {
      // 1. Auto-load /dataset.csv if local dataset is currently empty
      const { dataset: stored, legacyCleared } = loadDatasetFromStorage();
      if (legacyCleared && isMounted) {
        setUpgradeNotice('Upgraded to Sianor Dual-Hand (130 features). Legacy data was updated to ensure full compatibility.');
      }

      if (stored.length === 0) {
        const defaultData = await checkAndLoadDefaultDatasetCSV();
        if (defaultData && isMounted) {
          setDataset(defaultData.dataset);
          saveDatasetToStorage(defaultData.dataset);
          if (defaultData.labels.length > 0) {
            setLabels(defaultData.labels);
            saveLabelsToStorage(defaultData.labels);
          }
        }
      }

      // 2. Try restoring saved model from IndexedDB
      const mlp = new TFHandModel();
      const loadResult = await mlp.loadFromIndexedDB('sianor-hand-mlp');

      if (isMounted) {
        if (loadResult && loadResult.legacyCleared) {
          setUpgradeNotice('Upgraded to Sianor Dual-Hand (130 features). Legacy 63-input model was cleared.');
        } else if (loadResult === true) {
          setTfModel(mlp);
          setModelStatus({
            trained: true,
            modelType: 'mlp',
            labels: mlp.labels,
            accuracy: null,
          });
        }
      }
    }

    initializeDataAndModel();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="sianor-app">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelStatus={modelStatus}
        sampleCount={dataset.length}
      />

      {upgradeNotice && (
        <div className="upgrade-notice-banner">
          <div className="notice-left">
            <SparklesIcon size={16} />
            <span>{upgradeNotice}</span>
          </div>
          <button
            type="button"
            className="notice-dismiss-btn"
            onClick={() => setUpgradeNotice(null)}
          >
            Dismiss
          </button>
        </div>
      )}

      <main className="sianor-main-content">
        {activeTab === 'translate' && (
          <TranslateTab
            knnModel={knnModel}
            tfModel={tfModel}
            modelStatus={modelStatus}
            activeModelType={activeModelType}
            setActiveModelType={setActiveModelType}
            onGoToTrain={() => setActiveTab('train')}
          />
        )}

        {activeTab === 'teach' && (
          <TeachTab
            labels={labels}
            setLabels={setLabels}
            dataset={dataset}
            setDataset={setDataset}
          />
        )}

        {activeTab === 'train' && (
          <TrainTab
            dataset={dataset}
            labels={labels}
            setLabels={setLabels}
            knnModel={knnModel}
            setKnnModel={setKnnModel}
            tfModel={tfModel}
            setTfModel={setTfModel}
            modelStatus={modelStatus}
            setModelStatus={setModelStatus}
            onGoToTranslate={() => setActiveTab('translate')}
          />
        )}

        {activeTab === 'about' && (
          <AboutTab />
        )}
      </main>
    </div>
  );
}

export default App;
