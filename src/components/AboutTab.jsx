import React, { useState } from 'react';

export function AboutTab() {
  const [activeFaq, setActiveFaq] = useState(null);

  const vivaQuestions = [
    {
      q: 'Why translate landmark coordinates relative to the wrist (Landmark 0)?',
      a: 'If a user shifts slightly to the left, right, up, or down in front of the camera, raw pixel coordinates change drastically even though the hand shape is identical. By subtracting the wrist (x0, y0, z0) from every landmark, the hand skeleton is pinned to the origin (0, 0, 0), providing mathematical translation invariance.'
    },
    {
      q: 'Why scale landmarks by the maximum distance rather than a bounding box?',
      a: 'Bounding box dimensions (width & height) change when a hand rotates or changes aspect ratio. The maximum Euclidean distance from the wrist to any landmark gives an intrinsic biometric scale factor that is rotation-independent and scale-invariant, ensuring gestures are recognized whether you are 1 foot or 5 feet from the lens.'
    },
    {
      q: 'Why use a 15-frame rolling buffer with majority voting (>80% confidence)?',
      a: 'Real-time webcam video exhibits optical sensor noise, motion blur, and brief transitional hand movements between signs. Without temporal smoothing, the UI would flicker and trigger false speech announcements. Requiring at least 10 out of the last 15 frames to agree with >80% confidence ensures that only deliberate, stable gestures trigger speech.'
    },
    {
      q: 'Why compare a manual k-NN baseline against a TensorFlow.js Neural Network?',
      a: 'In academic machine learning projects, a baseline model (k-NN, k=5) provides an essential benchmark to prove that the deep learning Multilayer Perceptron actually learns superior generalized non-linear decision boundaries. Implementing k-NN from first principles also demonstrates deep algorithmic understanding without relying solely on black-box libraries.'
    },
    {
      q: 'What is a stratified train/test split and why is it crucial for gesture recognition?',
      a: 'A simple random split could accidentally put 90% of "Hello" samples in training and only 10% of "Stop" samples in test, causing severe class imbalance. A stratified split guarantees that every sign class is partitioned in an exact 80/20 proportion across training and test sets, yielding statistically sound, unbiased evaluation metrics.'
    },
    {
      q: 'How does Sianor run 100% in the browser with zero backend server dependencies?',
      a: 'MediaPipe Tasks Vision compiles Google hand tracking to WebAssembly (WASM) and WebGL shaders running directly inside the browser sandbox. TensorFlow.js utilizes WebGL/WebGPU to parallelize matrix multiplications directly on the client machine graphics hardware. The Web Speech API is natively built into modern browser engines.'
    }
  ];

  return (
    <div className="about-layout">
      {/* Overview Card */}
      <section className="sianor-card about-hero-card" aria-label="Project Overview">
        <div className="card-header-row">
          <div>
            <span className="card-kicker">Academic ML Mini-Project</span>
            <h2 className="card-title-lg">How Sianor Works</h2>
          </div>
          <span className="zero-backend-pill">100% Client-Side • Zero Backend</span>
        </div>

        <p className="about-intro-text">
          <strong>Sianor</strong> is an end-to-end machine learning system that translates human sign language
          into real-time written words and spoken voice. Built from the ground up for college ML curriculum,
          every stage of the pipeline—from computer vision landmark extraction to feature normalization,
          neural network training, evaluation, and speech synthesis—runs entirely within the user's web browser.
        </p>

        {/* Handcrafted SVG Pipeline Diagram */}
        <div className="pipeline-diagram-wrapper">
          <h3 className="subcard-title">End-to-End System Architecture</h3>
          <svg
            className="pipeline-svg"
            viewBox="0 0 940 210"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Background subtle connection rail */}
            <path
              d="M80 105 L860 105"
              stroke="#E2DAC9"
              strokeWidth="4"
              strokeDasharray="6 6"
            />

            {/* Block 1: Camera */}
            <g transform="translate(20, 45)">
              <rect width="125" height="110" rx="12" fill="#FAF7F2" stroke="#C85A32" strokeWidth="2.5" />
              <text x="62" y="38" textAnchor="middle" fill="#C85A32" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">1. WEBCAM</text>
              <text x="62" y="60" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">640×480 @ 60fps</text>
              <text x="62" y="78" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">HTML5 Video</text>
              <circle cx="62" cy="94" r="4" fill="#C85A32" />
            </g>

            {/* Arrow 1 */}
            <path d="M150 105 L175 105" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 2: MediaPipe */}
            <g transform="translate(180, 45)">
              <rect width="135" height="110" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="67" y="38" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">2. VISION</text>
              <text x="67" y="60" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">MediaPipe Hands</text>
              <text x="67" y="78" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">21 3D Landmarks</text>
              <text x="67" y="94" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">WASM / GPU</text>
            </g>

            {/* Arrow 2 */}
            <path d="M320 105 L345 105" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 3: Preprocessing */}
            <g transform="translate(350, 45)">
              <rect width="145" height="110" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="72" y="38" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">3. NORMALIZE</text>
              <text x="72" y="58" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• Wrist to (0,0,0)</text>
              <text x="72" y="73" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• Scale by Max Dist</text>
              <text x="72" y="88" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• Flip Left to Right</text>
              <text x="72" y="101" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">63-D Vector</text>
            </g>

            {/* Arrow 3 */}
            <path d="M500 105 L525 105" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 4: Inference */}
            <g transform="translate(530, 45)">
              <rect width="135" height="110" rx="12" fill="#FAF7F2" stroke="#E09F3E" strokeWidth="2.5" />
              <text x="67" y="38" textAnchor="middle" fill="#E09F3E" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">4. CLASSIFIER</text>
              <text x="67" y="60" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Deep MLP (TF.js)</text>
              <text x="67" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">63 → 128 → 64 → K</text>
              <text x="67" y="93" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">or k-NN (k=5)</text>
            </g>

            {/* Arrow 4 */}
            <path d="M670 105 L695 105" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 5: Temporal Filter */}
            <g transform="translate(700, 45)">
              <rect width="115" height="110" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="57" y="38" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">5. SMOOTH</text>
              <text x="57" y="60" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Majority Vote</text>
              <text x="57" y="78" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">10 / 15 Frames</text>
              <text x="57" y="94" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">&gt; 0.80 Conf</text>
            </g>

            {/* Arrow 5 */}
            <path d="M820 105 L840 105" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 6: Output */}
            <g transform="translate(845, 45)">
              <rect width="85" height="110" rx="12" fill="#FAF7F2" stroke="#C85A32" strokeWidth="2.5" />
              <text x="42" y="38" textAnchor="middle" fill="#C85A32" fontWeight="bold" fontSize="12" fontFamily="'Fraunces', serif">6. VOICE</text>
              <text x="42" y="60" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Sentence</text>
              <text x="42" y="78" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">Speech</text>
              <text x="42" y="94" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Synthesis</text>
            </g>

            {/* Marker definitions */}
            <defs>
              <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                <polygon points="0 0, 7 3.5, 0 7" fill="#C85A32" />
              </marker>
            </defs>
          </svg>
        </div>
      </section>

      {/* College Viva Questions & Answers Accordion */}
      <section className="sianor-card viva-card" aria-label="Viva Preparation">
        <div className="card-header-row">
          <div>
            <span className="card-kicker">Viva Voce Preparation</span>
            <h3 className="card-title-sm">College ML Mini-Project Q&A Guide</h3>
          </div>
          <span className="exam-tag">High-Yield Viva Concepts</span>
        </div>

        <p className="card-instruction">
          Click any question below to inspect the mathematical and architectural justifications.
          These represent standard questions asked by faculty evaluators during project reviews.
        </p>

        <div className="viva-accordion">
          {vivaQuestions.map((item, idx) => {
            const isOpen = activeFaq === idx;
            return (
              <div
                key={idx}
                className={`viva-item ${isOpen ? 'viva-item-open' : ''}`}
                onClick={() => setActiveFaq(isOpen ? null : idx)}
              >
                <div className="viva-header">
                  <span className="viva-q-prefix">Q{idx + 1}:</span>
                  <h4 className="viva-question-text">{item.q}</h4>
                  <span className="viva-toggle-icon">{isOpen ? '−' : '+'}</span>
                </div>
                {isOpen && (
                  <div className="viva-answer-content">
                    <p>{item.a}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
