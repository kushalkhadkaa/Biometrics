# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-06-08

### Added
- **Core Forensic Pipeline** (`fingerprint_pipeline.py`):
  - Robust intensity range rescaling and automatic polarity checking (dark/light ridge detection).
  - uneven illumination correction using high-pass Gaussian filtering.
  - Local contrast stretching using CLAHE ($8 \times 8$ grid, clip limit $2.0$).
  - Region of Interest (ROI) texture-gradient segmentation using Otsu binarization and morphologically smoothed convex hulls.
  - Structure Tensor local orientation and structural coherence estimation.
  - Gabor enhancement convolved at local orientation angles with coherence-guided blending.
- **Parallel Masking Engine**:
  - Color Saturation analysis in HSV space for pen/stamp isolation.
  - Probabilistic Hough Line Transform mapping for straight ruled lines.
  - Adaptive component segmentation and local-median subtraction for dark print text and handwritten ink strokes.
  - Coordinate mapping for manual rectangle, line, and polygonal override paths.
- **Guarded Reconstruction System**:
  - Size-bounded Telea inpainting (restricted to components $\le 1600\text{ px}^2$ and bounding boxes $\le 48 \times 48$).
  - Neighborhood ridge support verification (minimum $55\%$ clean ridge density required).
  - Automatic blocking and warning system for unsafe regions to prevent false evidence fabrication.
- **Interactive Web App Console** (`web_app.py`):
  - Multi-threaded job queue system for asynchronous background processing.
  - Interactive dashboard demonstrating side-by-side original, clean output, and three-color provenance maps.
  - Diagnostic charts showing intermediate algorithm stages.
  - Case history manager utilizing persistent JSON storage.
  - Automated PDF report compilation with metrics and comparison maps using Matplotlib.
- **Project Structure**:
  - Standard PEP 517/621-compliant `pyproject.toml` package definition.
  - Diagnostic Jupyter Workspace (`notebooks/Fingerprint_Forensic_Preprocessing_Reconstruction.ipynb`).
  - Industry-grade `README.md` with system flowcharts and mathematical documentation.
  - PowerShell startup scripts (`run_ui.ps1`, `run_notebook.ps1`).
