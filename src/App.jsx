import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { TranslateTab } from './components/TranslateTab';
import { TeachTab } from './components/TeachTab';
import { TrainTab } from './components/TrainTab';
import { AboutTab } from './components/AboutTab';

import { DEFAULT_SIGN_LABELS } from './ml/demoData';
import {
  loadDatasetFromStorage,
  loadLabelsFromStorage,
} from './ml/storage';
import { KNNClassifier } from './ml/knn';
import { TFHandModel } from './ml/tfModel';
import './styles/sianor.css';

export function App() {
  const [activeTab, setActiveTab] = useState('translate');

  // Signs vocabulary
  const [labels, setLabels] = useState(() => loadLabelsFromStorage(DEFAULT_SIGN_LABELS));

  // Dataset of landmark samples: Array<{ label: string, features: Float32Array }>
  const [dataset, setDataset] = useState(() => loadDatasetFromStorage());

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

  // On mount: try restoring saved model from IndexedDB
  useEffect(() => {
    let isMounted = true;

    async function restoreSavedModel() {
      const mlp = new TFHandModel();
      const loaded = await mlp.loadFromIndexedDB('sianor-hand-mlp');

      if (loaded && isMounted) {
        setTfModel(mlp);
        setModelStatus({
          trained: true,
          modelType: 'mlp',
          labels: mlp.labels,
          accuracy: null,
        });
      }
    }

    restoreSavedModel();

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
