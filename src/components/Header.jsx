import React from 'react';
import { HandIcon, BookIcon, BrainIcon, InfoIcon } from './Icons';

export function Header({ activeTab, setActiveTab, modelStatus, sampleCount }) {
  const tabs = [
    { id: 'translate', label: 'Translate', icon: HandIcon, count: null },
    { id: 'teach', label: 'Teach', icon: BookIcon, count: sampleCount > 0 ? `${sampleCount}` : null },
    { id: 'train', label: 'Train', icon: BrainIcon, count: modelStatus.trained ? 'Ready' : null },
    { id: 'about', label: 'About', icon: InfoIcon, count: null },
  ];

  return (
    <header className="sianor-header">
      <div className="header-top">
        <div className="brand-group">
          <div className="brand-logo-wrap">
            <span className="brand-title">Sianor</span>
            {/* Hand-drawn organic squiggle underline */}
            <svg className="brand-squiggle" viewBox="0 0 140 14" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M3 9C24 4 48 11 72 7C95 3 118 10 137 6"
                stroke="#C85A32"
                strokeWidth="3.2"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <p className="brand-tagline">Real-time sign language translator & ML lab</p>
        </div>

        <div className="header-status-badge">
          <span className={`status-dot ${modelStatus.trained ? 'status-active' : 'status-waiting'}`} />
          <span className="status-label">
            {modelStatus.trained
              ? `Model active (${modelStatus.modelType.toUpperCase()})`
              : 'No trained model (visit Teach / Train)'}
          </span>
        </div>
      </div>

      <nav className="tab-nav" aria-label="Main Navigation">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              className={`tab-btn ${isActive ? 'tab-btn-active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="tab-icon" size={18} />
              <span className="tab-label">{tab.label}</span>
              {tab.count && (
                <span className={`tab-badge ${isActive ? 'tab-badge-active' : ''}`}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
