# Contributing to Fingerprint-Image-Enhancement-and-Reconstruction

First off, thank you for taking the time to contribute! Contributions from the global open-source community help make this forensic tool more robust, performant, and secure.

To maintain professional standards, please review the following guidelines before submitting code.

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to Kushal Khadka.

---

## Developer Guide & Environment Setup

### 1. Fork and Clone
1. Fork the repository on GitHub: `https://github.com/kushalkhadkaa/Biometrics`
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Biometrics.git
   cd Biometrics
   ```
3. Set the upstream remote:
   ```bash
   git remote add upstream https://github.com/kushalkhadkaa/Biometrics.git
   ```

### 2. Dependency Management
Initialize a virtual environment and install dependencies in development mode:
```bash
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest
```

---

## Code Quality Standards

To maintain a clean, readable codebase, we enforce formatting, linting, and type checking:

### 1. Code Formatting (PEP 8)
All Python code is formatted using `black` with a line-length limit of 100 characters. Run formatting before committing:
```bash
black --line-length 100 fingerprint_pipeline.py web_app.py
```

### 2. Linting
Verify code styling and check for unused imports or variables using `flake8`:
```bash
flake8 --max-line-length 100 --ignore=E203,W503 fingerprint_pipeline.py web_app.py
```

### 3. Static Type Analysis
We utilize static type annotations to ensure mathematical and image arrays are processed cleanly. Run type checks using `mypy`:
```bash
mypy --ignore-missing-imports fingerprint_pipeline.py web_app.py
```

---

## Project Governance & Maintainer Rights Reservation

This project is authored and maintained by **Kushal Khadka**. To protect the scientific integrity and future direction of the pipeline, the following reservations are established:

### 1. Architectural Authority
- The primary author reserves the absolute right to direct the development roadmap and approve or reject any architectural changes, algorithm updates, or configuration thresholds.
- Contributions that alter core forensic validation rules (such as the guarded inpainting support ratios or size limits) will undergo rigorous scientific review prior to integration.

### 2. Licensing & Attribution
- By submitting a Pull Request, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
- Historical attribution to the original author (Kushal Khadka) must be maintained in all forks, distributions, and downstream packages.

---

## Submission Process

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Commit**:
   Commit your changes using semantic prefixes (e.g. `feat: add adaptive saturation`, `fix: correct Gabor scale`).
3. **Verify Locally**:
   Ensure your modifications compile and pass syntax validation:
   ```bash
   python -m py_compile web_app.py fingerprint_pipeline.py
   ```
4. **Push and Open Pull Request**:
   Push to your fork and create a Pull Request targeting the upstream `main` branch. Provide a detailed description of your modifications and validation results.
