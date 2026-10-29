import React, { useState } from 'react';

export function AboutTab() {
  const [activeFaq, setActiveFaq] = useState(null);

  const vivaQuestions = [
    {
      q: 'How does Sianor support both one-hand and two-hand signs within a single 130-feature vector?',
      a: 'The feature vector is partitioned into 4 distinct regions: Slot 1 (indices 0..62) holds the 63 normalized coordinates of Hand 1, Slot 2 (indices 63..125) holds the 63 coordinates of Hand 2, indices 126..128 hold the 3D displacement vector between the two wrists (scaled by average hand size), and index 129 holds a binary indicator (1.0 = both hands visible, 0.0 = single hand). For one-hand gestures, Slot 1 contains the active hand, while Slot 2, the offset, and the flag are cleanly zero-padded.'
    },
    {
      q: 'Why order detected hands by their horizontal screen position (leftmost wrist first)?',
      a: 'If a neural network received hands in random or fluctuating order frame-to-frame, the same physical two-handed sign would produce swapped feature slots, confusing the classifier. Sorting hands by their screen wrist X-position ensures deterministic slot assignment: Slot 1 always represents the leftmost hand on screen, and Slot 2 represents the rightmost hand.'
    },
    {
      q: 'Why include the 3D inter-wrist offset scaled by average hand size?',
      a: 'Individual hands can have identical internal shapes for different signs (e.g. flat palms), but the relative distance and orientation between the two hands distinguish signs like "Please" (palms touching near chest), "Thank You" (both hands moving outward together), and "Book" (hands opening side-by-side). Dividing the raw displacement by the average hand scale ensures the inter-hand spatial relationship remains scale-invariant regardless of distance from the camera.'
    },
    {
      q: 'How does dual-hand data augmentation work with slot swapping?',
      a: 'To make the model robust against user positioning and slight posture changes, augmentation applies in-plane rotation and Gaussian jitter to both hand slots and the offset vector. Crucially, when horizontal mirroring is simulated, both hand coordinates have their X-values negated, the horizontal offset dx is inverted, and Slot 1 and Slot 2 are swapped, accurately reflecting the mirrored geometry.'
    },
    {
      q: 'Why translate each hand landmark relative to its own wrist (Landmark 0)?',
      a: 'If a user shifts slightly in front of the camera, raw pixel coordinates change drastically even though the hand shape is unchanged. By independently subtracting each hand’s wrist position (x0, y0, z0) from its 21 landmarks, the hand skeleton is pinned to the coordinate origin (0, 0, 0), achieving rigorous translation invariance.'
    },
    {
      q: 'Why use a 15-frame rolling buffer with majority voting (>80% confidence)?',
      a: 'Real-time webcam video exhibits optical noise, motion blur, and brief transitional hand movements between signs. Without temporal smoothing, the UI would flicker and trigger false speech announcements. Requiring at least 10 out of the last 15 frames to agree with >80% confidence ensures that only deliberate, stable gestures trigger speech.'
    },
    {
      q: 'How does Sianor run 100% in the browser with zero backend server dependencies?',
      a: 'MediaPipe Tasks Vision compiles Google hand tracking to WebAssembly (WASM) and WebGL shaders running directly inside the browser sandbox. TensorFlow.js utilizes WebGL/WebGPU to parallelize matrix multiplications directly on client hardware. The Web Speech API is natively built into modern browser engines, ensuring total privacy and zero server latency.'
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
          <span className="zero-backend-pill">100% Client-Side • Dual-Hand ML</span>
        </div>

        <p className="about-intro-text">
          <strong>Sianor</strong> is an end-to-end machine learning system that translates human sign language
          into real-time written words and spoken voice. Built from the ground up for college ML curriculum,
          Sianor supports both <strong>one-hand and two-hand signs</strong> through an invariant 130-feature
          biometric vector, dual-hand MediaPipe tracking, client-side neural network training, and native speech synthesis.
        </p>

        {/* Handcrafted SVG Pipeline Diagram */}
        <div className="pipeline-diagram-wrapper">
          <h3 className="subcard-title">Dual-Hand System Architecture</h3>
          <svg
            className="pipeline-svg"
            viewBox="0 0 940 215"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Background connection rail */}
            <path
              d="M80 108 L860 108"
              stroke="#E2DAC9"
              strokeWidth="4"
              strokeDasharray="6 6"
            />

            {/* Block 1: Camera */}
            <g transform="translate(15, 45)">
              <rect width="130" height="115" rx="12" fill="#FAF7F2" stroke="#C85A32" strokeWidth="2.5" />
              <text x="65" y="36" textAnchor="middle" fill="#C85A32" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">1. WEBCAM</text>
              <text x="65" y="58" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">640×480 @ 60fps</text>
              <text x="65" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">HTML5 Video</text>
              <text x="65" y="94" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">1 or 2 Hands</text>
            </g>

            {/* Arrow 1 */}
            <path d="M150 108 L175 108" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 2: MediaPipe */}
            <g transform="translate(180, 45)">
              <rect width="140" height="115" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="70" y="36" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">2. VISION</text>
              <text x="70" y="58" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">MediaPipe Hands</text>
              <text x="70" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">numHands: 2</text>
              <text x="70" y="94" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">42 3D Keypoints</text>
            </g>

            {/* Arrow 2 */}
            <path d="M325 108 L350 108" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 3: Preprocessing */}
            <g transform="translate(355, 45)">
              <rect width="155" height="115" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="77" y="34" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">3. NORMALIZE</text>
              <text x="77" y="54" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• Slot 1: Hand 1 (63)</text>
              <text x="77" y="69" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• Slot 2: Hand 2 (63)</text>
              <text x="77" y="84" textAnchor="middle" fill="#1F1B18" fontSize="10" fontFamily="'DM Sans', sans-serif">• 3D Wrist Offset (3)</text>
              <text x="77" y="100" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="700" fontFamily="'DM Sans', sans-serif">130-D Vector</text>
            </g>

            {/* Arrow 3 */}
            <path d="M515 108 L540 108" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 4: Inference */}
            <g transform="translate(545, 45)">
              <rect width="135" height="115" rx="12" fill="#FAF7F2" stroke="#E09F3E" strokeWidth="2.5" />
              <text x="67" y="36" textAnchor="middle" fill="#E09F3E" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">4. CLASSIFIER</text>
              <text x="67" y="58" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Deep MLP (TF.js)</text>
              <text x="67" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">130 → 128 → 64 → K</text>
              <text x="67" y="94" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">or k-NN (k=5)</text>
            </g>

            {/* Arrow 4 */}
            <path d="M685 108 L710 108" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 5: Temporal Filter */}
            <g transform="translate(715, 45)">
              <rect width="115" height="115" rx="12" fill="#FAF7F2" stroke="#292524" strokeWidth="2" />
              <text x="57" y="36" textAnchor="middle" fill="#1F1B18" fontWeight="bold" fontSize="13" fontFamily="'Fraunces', serif">5. SMOOTH</text>
              <text x="57" y="58" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Majority Vote</text>
              <text x="57" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">10 / 15 Frames</text>
              <text x="57" y="94" textAnchor="middle" fill="#C85A32" fontSize="9" fontWeight="600" fontFamily="'DM Sans', sans-serif">&gt; 0.80 Conf</text>
            </g>

            {/* Arrow 5 */}
            <path d="M835 108 L855 108" stroke="#C85A32" strokeWidth="2.5" markerEnd="url(#arrowhead)" />

            {/* Block 6: Output */}
            <g transform="translate(860, 45)">
              <rect width="80" height="115" rx="12" fill="#FAF7F2" stroke="#C85A32" strokeWidth="2.5" />
              <text x="40" y="36" textAnchor="middle" fill="#C85A32" fontWeight="bold" fontSize="12" fontFamily="'Fraunces', serif">6. VOICE</text>
              <text x="40" y="58" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Sentence</text>
              <text x="40" y="76" textAnchor="middle" fill="#7C7267" fontSize="10" fontFamily="'DM Sans', sans-serif">Speech</text>
              <text x="40" y="94" textAnchor="middle" fill="#1F1B18" fontSize="11" fontFamily="'DM Sans', sans-serif">Synthesis</text>
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
