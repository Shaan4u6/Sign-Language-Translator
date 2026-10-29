# Sianor — In-Browser Sign Language Translator & ML Lab

**Sianor** is a client-side machine learning application that translates real-time hand gestures into text and spoken voice entirely within the browser. Designed as an academic ML mini-project, Sianor exposes the end-to-end machine learning lifecycle directly in the interface: from computer vision keypoint extraction and feature engineering to model training, evaluation metrics, and speech synthesis.

---

## 🌟 Key Features

1. **Zero-Backend Client-Side Execution**:
   - Runs 100% in the browser sandbox. No Python backend, no Flask/FastAPI server, no external API keys, and zero cloud data transmission.
2. **Real-Time Hand Landmark Tracking**:
   - MediaPipe Hands (`@mediapipe/tasks-vision`) extracts 21 three-dimensional anatomical hand landmarks from standard 60 FPS webcam feeds.
3. **Rigorous Invariant Feature Preprocessing**:
   - Converts raw video pixels into a normalized, translation-invariant, scale-invariant, and handedness-invariant 63-dimensional feature vector.
4. **Dual Model Architecture**:
   - **Baseline**: Manual $k$-Nearest Neighbors ($k$-NN, $k=5$) implemented from first principles using Euclidean distance and voting.
   - **Deep Learning**: TensorFlow.js Multilayer Perceptron (MLP) with Dense layers, ReLU activations, Dropout (0.30), Adam optimization, and Softmax categorical cross-entropy.
5. **Real-Time Temporal Smoothing**:
   - 15-frame rolling buffer requiring $\ge 10$ agreeing frames with $>80\%$ confidence before accepting a gesture. Prevents optical flicker and accidental triggering.
6. **Sentence Assembly & Speech Synthesis**:
   - Automatically constructs words into coherent phrases. Speaks output using the native Web Speech API (`speechSynthesis`).
7. **Comprehensive Evaluation Suite**:
   - Stratified 80/20 train/test split.
   - Real-time Canvas loss and accuracy curves per epoch.
   - Side-by-side $k$-NN vs. MLP test accuracy.
   - Per-class Precision, Recall, and F1-score tables.
   - Interactive $N \times N$ Confusion Matrix heatmap.
8. **Persistence & Data Portability**:
   - Export and import gesture datasets via standard CSV.
   - Saves trained model weights directly to browser IndexedDB for automatic reload.
   - Includes a **Load Demo Dataset** option (360 calibrated samples across 8 default gestures) for instant evaluation and demonstration.
9. **Warm Editorial Paper Aesthetic**:
   - Inspired by printed books and artisanal lab notebooks: warm paper background (`#FAF7F2`), deep ink typography (`#1F1B18`), terracotta (`#C85A32`) and marigold (`#E09F3E`) accents, Fraunces serif headings, hand-drawn SVG squiggles, and polaroid-framed video feed.

---

## 🛠 Tech Stack

- **Framework**: React 19 + Vite (Fast HMR and production bundling)
- **Computer Vision**: `@mediapipe/tasks-vision` (WebAssembly & WebGL hand landmarker)
- **Machine Learning**: `@tensorflow/tfjs` (Client-side GPU/WASM tensor training & inference)
- **Speech Engine**: Web Speech API (`window.speechSynthesis`)
- **Dataset Storage**: LocalStorage (samples and labels) & IndexedDB (TensorFlow.js model artifacts)
- **Typography**: Fraunces (Headings), DM Sans (UI), Caveat (Handwritten captions)

---

## 📐 Machine Learning Pipeline

```
[Webcam Stream] ──> [MediaPipe Tasks Vision (21 Landmarks)]
                            │
                            ▼
        [Mathematical Invariant Preprocessing]
         1. Wrist Translation: P_i - P_wrist
         2. Max-Distance Scaling: P_i / max_dist
         3. Handedness Reflection: X = -X if Left
         4. Flattening: [x0, y0, z0, ..., x20, y20, z20] (63-D)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [k-NN Baseline (k=5)]       [TensorFlow.js Deep MLP]
     Euclidean distance           63 -> Dense 128 (ReLU)
     Vote tallying                -> Dropout 0.3
                                  -> Dense 64 (ReLU)
                                  -> Dense K (Softmax)
            └───────────────┬───────────────┘
                            ▼
       [Temporal Smoothing (15-Frame Rolling Window)]
       Condition: Majority vote >= 10 frames & Conf > 0.80
                            │
                            ▼
              [Sentence Assembly & Web Speech API]
```

### 1. Preprocessing Details (`src/ml/preprocessor.js`)
- **Wrist Origin Centering**: Landmark 0 (wrist) is translated to $(0, 0, 0)$ via $\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0$. This removes screen position dependence.
- **Biometric Scale Invariance**: We calculate $d_{\max} = \max_{i} \|\mathbf{p}'_i\|_2$ and divide all coordinates by $d_{\max}$. The gesture representation remains constant whether the user sits 1 foot or 6 feet away.
- **Handedness Reflection**: If MediaPipe detects a left hand, the X-coordinates are reflected ($x \leftarrow -x$). This allows right-hand models to accept left-handed gestures out of the box.
- **63-Feature Vector**: The 21 points $\times$ 3 coordinates $(x, y, z)$ are flattened into a contiguous 1D array of 63 floating-point numbers.

### 2. Data Augmentation
To improve generalizability on small collected datasets, synthetic samples can be generated by applying:
- In-plane rotation around the $Z$-axis: $\theta \in [-12^\circ, +12^\circ]$
- Biometric scale jitter: $s \in [0.92, 1.08]$
- Gaussian coordinate noise: $\mathcal{N}(0, 0.015)$

---

## 🚀 Quickstart & Setup

### Prerequisites
- Node.js $\ge 18.0.0$ (Node v22 recommended)
- A modern browser with webcam support (Chrome, Edge, Firefox, or Safari)

### Installation
```bash
# Clone or navigate to the project directory
cd d:/sign_language

# Install project dependencies
npm install

# Start the local development server
npm run dev
```

Open the printed URL (typically `http://localhost:5173`) in your browser and grant webcam permissions when prompted.

### Production Build
```bash
npm run build
npm run preview
```

---

## 📖 Application Workflow

### 1. Translate Tab (Home)
- **Live Polaroid Feed**: Displays live webcam stream with overlaid 21-joint skeleton and fingertip tracking beads.
- **Confidence Meter**: Visualizes live prediction certainty with an explicit 80% threshold indicator.
- **Majority Vote Smoothing**: A prediction is locked only when the same sign is detected in at least 10 of the last 15 frames with $>80\%$ confidence.
- **Sentence Assembly**: Words are appended to the sentence without duplicates while the hand is held in frame. Includes **Speak**, **Undo**, and **Clear** controls.
- **Auto-Speak Toggle**: Automatically reads signs aloud as soon as they stabilize.
- **Model Switcher**: Toggle real-time inference between the Deep Neural Network (MLP) and the $k$-NN baseline.

### 2. Teach Tab (Data Collection)
- **Sign Vocabulary**: Add, rename, or delete sign categories (default: *Hello, Thank You, Yes, No, I Love You, Stop, Please, OK*).
- **10-Second Recording**: Click **Record** on any sign. A 3-second preparation countdown is followed by 10 seconds of continuous frame capture (~200 samples).
- **CSV Portability**: Export your collected samples as CSV or import pre-existing datasets.
- **Load Demo Dataset**: Single-click button that instantly loads 360 pre-calibrated landmark samples across all 8 standard signs.

### 3. Train Tab (Model Laboratory)
- **Preprocessing Overview**: Educational cards detailing the 4 mathematical transformations.
- **Augmentation Switch**: Enable or disable rotational, scale, and noise augmentation.
- **Train Launch**: Trains both $k$-NN ($k=5$) and TF.js MLP (50 epochs) on an 80/20 stratified split.
- **Live Loss & Accuracy Charts**: Real-time canvas graphs plotting training and validation metrics per epoch.
- **Evaluation Dashboard**:
  - Side-by-side test accuracy comparison cards.
  - Per-class Precision, Recall, and F1-score table with support count.
  - Interactive Confusion Matrix heatmap showing actual vs. predicted classifications.
  - Automatic model persistence to IndexedDB (`indexeddb://sianor-hand-mlp`).

### 4. About Tab (Architecture & Viva Voce Guide)
- System architecture diagram (SVG).
- Step-by-step pipeline narrative.
- Frequently asked college viva questions with academic answers.

---

## 🎓 College Viva Voce Questions & Answers

### Q1: Why normalize landmark coordinates relative to the wrist?
**Answer**: In video capture, the user may sit on the left, right, top, or bottom of the screen. Subtracting the wrist position $(x_0, y_0, z_0)$ from all 21 landmarks anchors the hand to $(0, 0, 0)$. This gives **translation invariance**, ensuring identical numerical values regardless of hand position in the camera frame.

### Q2: Why scale by maximum Euclidean distance instead of a bounding box?
**Answer**: Bounding box width and height change as the hand rotates in 3D space, introducing distortion. The maximum Euclidean distance from the wrist to any landmark represents an intrinsic biological ray that remains stable under in-plane rotation, providing true **scale invariance**.

### Q3: Why is a stratified train/test split necessary?
**Answer**: Gesture datasets are often collected unevenly. A standard random split might assign 90% of samples from one class to training and only 10% from another to testing, biasing evaluation. A stratified split ensures every class is partitioned in an exact 80/20 ratio, guaranteeing balanced, statistically representative test metrics.

### Q4: Why use a 15-frame buffer with majority voting?
**Answer**: Real-time optical video is subject to camera sensor noise, hand tremors, and transitional gestures when moving between signs. Requiring at least 10 out of the last 15 frames to agree with $>80\%$ confidence filters out spurious transient detections and prevents speech synthesis spam.

### Q5: How does the $k$-NN baseline differ from the MLP?
**Answer**: $k$-NN is a non-parametric, lazy learner that calculates Euclidean distances across the entire training dataset at query time. The MLP is a parametric deep learning model that learns non-linear feature abstractions via backpropagation and weights. Comparing both demonstrates whether deep hierarchical representations yield superior decision boundaries over raw distance matching.

### Q6: How does the application run without a backend server?
**Answer**: Modern web standards allow high-performance client-side computation:
1. MediaPipe uses WebAssembly (WASM) and WebGL shaders for real-time hand detection.
2. TensorFlow.js leverages WebGL/WebGPU to parallelize matrix multiplications directly on the client's GPU.
3. The Web Speech API is built natively into modern browser engines.
4. IndexedDB and LocalStorage provide persistent on-device client storage.

---

## 📁 Project Structure

```
d:/sign_language/
├── public/
│   ├── mediapipe/wasm/           # Bundled MediaPipe Vision WASM binaries
│   └── models/                   # Local MediaPipe hand_landmarker.task model
├── src/
│   ├── ml/
│   │   ├── mediapipe.js          # Vision task loader & canvas skeleton renderer
│   │   ├── preprocessor.js       # Wrist translation, distance scaling, reflection, augmentation
│   │   ├── knn.js                # Manual k-NN classifier (k=5, Euclidean distance)
│   │   ├── tfModel.js            # TensorFlow.js MLP (Dense 128 -> Dropout 0.3 -> Dense 64 -> Softmax)
│   │   ├── metrics.js            # Stratified split, precision, recall, F1, confusion matrix
│   │   ├── demoData.js           # 8-class calibrated demo dataset generator
│   │   └── storage.js            # LocalStorage & IndexedDB dataset & model persistence
│   ├── components/
│   │   ├── Header.jsx            # Branding with organic squiggle & navigation tabs
│   │   ├── CameraView.jsx        # Polaroid-styled camera frame & canvas overlay
│   │   ├── TranslateTab.jsx      # Real-time inference, majority voting, TTS sentence builder
│   │   ├── TeachTab.jsx          # Vocabulary management, 10s recording, CSV import/export
│   │   ├── TrainTab.jsx          # Preprocessing guide, live training charts, evaluation dashboard
│   │   ├── TrainingCharts.jsx    # Real-time HTML5 Canvas loss and accuracy curves
│   │   ├── ConfusionMatrix.jsx   # Interactive multi-class heatmap
│   │   ├── AboutTab.jsx          # Architecture diagram & viva preparation cards
│   │   └── Icons.jsx             # Handcrafted SVG icons
│   ├── styles/
│   │   └── sianor.css            # Warm paper design system & responsive layout
│   ├── App.jsx                   # Main application state orchestrator
│   ├── main.jsx                  # React application entry point
│   └── index.css                 # Base theme import
├── index.html                    # HTML template with Google Fonts (Fraunces, DM Sans, Caveat)
├── package.json                  # Dependencies & scripts
└── README.md                     # Comprehensive academic documentation
```

---

## 📜 License & Academic Attribution
Developed as an open-source educational machine learning project for college coursework and demonstration.
