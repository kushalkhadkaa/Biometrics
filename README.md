# Biometrics: Forensic Fingerprint Preprocessing & Reconstruction

An advanced, end-to-end Python pipeline and web application designed for conservative forensic fingerprint preprocessing, artifact removal, ridge flow reconstruction, and noise suppression. 

This system is built to process latent or forensic fingerprint images containing non-ridge artifacts—such as handwritten annotations, pen marks, scan lines, border frames, and localized noise—while preserving and reconstructing the underlying ridge flow with guarded validation.

---

## Key Features

- **Automated Algorithmic Pipeline** (`fingerprint_pipeline.py`):
  1. **Contrast & Illumination Normalization**: Combines global polarity checks, robust percentile scaling, and adaptive illumination correction.
  2. **Local Contrast Enhancement**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for ridge definition.
  3. **Texture-Based ROI Segmentation**: Isolates the fingerprint from the background using local standard deviation, Sobel gradients, and Otsu thresholding.
  4. **Ridge-Preserving Denoising**: Employs bilateral and median filtering to eliminate high-frequency sensor noise without blurring ridges.
  5. **Advanced Artifact Masking**: Automated detection of color markings, straight ruled lines (via Hough Line Transform), dark text components, and thick ink strokes. Supports manual polygonal/linear override regions.
  6. **Guarded Inpainting & Reconstruction**: Reconstructs damaged ridges using local coherence-guided Telea inpainting. Gaps with insufficient neighborhood support remain blocked to prevent the creation of false evidence.
  7. **Orientation-Guided Gabor Enhancement**: Enhances ridge flow clarity by applying orientation-selective Gabor filters matching the local ridge period.
  8. **Comprehensive Quality Metrics**: Evaluates ROI coverage, contrast STD, mean ridge coherence, Laplacian variance, noise residual (MAD), and artifact ratios.

- **Interactive Forensic Web Console** (`web_app.py`):
  - **Live Pipeline Visualization**: Inspect intermediate outputs (normalization, ROI, denoise, masks, provenance, and final enhancement).
  - **Scientific & Stakeholder Reports**: Displays real-time quality assessments and tailored warning flags based on statistical limits.
  - **Interactive Diagnostics**: Inspect pixel-level details and download high-resolution output images or comprehensive PDF reports.
  - **Case History**: View logs and records of previously analyzed cases.

---

## Project Structure

```
d:/fingerprint/
├── fingerprint_pipeline.py  # Core image processing algorithms and pipeline
├── web_app.py               # Flask-based web server and interactive dashboard
├── requirements.txt         # Package dependencies
├── run_ui.ps1               # PowerShell script to run the web application
├── run_notebook.ps1         # PowerShell script to run the Jupyter environment
├── scripts/
│   └── build_notebook.py    # Helper script to compile notebooks
├── notebooks/
│   └── Fingerprint_Forensic_Preprocessing_Reconstruction.ipynb
├── utils/                   # Utility scripts, assets, and sample images
└── outputs/                 # Output directories for processed cases (ignored in git)
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kushalkhadkaa/Biometrics.git
   cd Biometrics
   ```

2. **Set up a virtual environment (Recommended)**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launching the Web Console
You can quickly run the Flask web app using the provided PowerShell script or directly using Python:
```powershell
./run_ui.ps1
```
Or:
```bash
python web_app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to start processing images.

### 2. Jupyter Notebook Environment
To explore the pipeline step-by-step or run manual experiments:
```powershell
./run_notebook.ps1
```

---

## Scientific Considerations & Verification
- **Guarded Inpainting**: Gaps containing removed artifacts are only reconstructed if there is high ridge coherence and structural support. If a gap is too large (e.g., width > 48px or area > 1600px), it is flagged as **Blocked Artifact Pixels** to maintain forensic integrity and avoid generating imaginary ridge patterns.
- **Provenance Mapping**: The pipeline generates a three-color overlay to transparently report output integrity:
  - <span style="color:#0ea5e9">■</span> **Blue (Reconstructed)**: Gaps successfully inpainted.
  - <span style="color:#ef4444">■</span> **Red (Blocked)**: Artifacts removed but left empty due to size/coherence limits.
  - <span style="color:#eab308">■</span> **Yellow (Review)**: Ambiguous zones recommended for manual inspection.
