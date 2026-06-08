# Fingerprint-Image-Enhancement-and-Reconstruction: Latent Preprocessing Pipeline

Fingerprint-Image-Enhancement-and-Reconstruction is an industry-grade, scientifically bounded image processing suite and interactive workspace designed to isolate, clean, and reconstruct latent or noisy fingerprint evidence. 

By employing deterministic computer vision algorithms, the pipeline removes non-ridge artifacts—such as handwritten annotations, pen strokes, ruler markings, scanner crop lines, and scanner noise—while strictly preserving and validating the integrity of the underlying ridge flow.

---

## Technical Architecture & Processing Pipeline

The following flowchart details the multi-stage computational graph. The pipeline is structured to ensure that every operation is verifiable, mathematically bounded, and auditable.

```mermaid
flowchart TD
    %% System Input
    Input[Raw Evidence Image\nRGB / PNG / JPEG / BMP] --> Load[1. Load & EXIF Transpose\nImageOps.exif_transpose]
    Load --> Gray[2. Grayscale Conversion\nRGB to float32 range 0.0, 1.0]
    
    %% Normalization & Polarity
    Gray --> Stretch[3. Robust Rescale\n1.0 to 99.0 Percentile Stretch]
    Stretch --> Polarity{4. Polarity Check\nBorder Median vs Center Median}
    Polarity -- Background Dark --> Invert[Invert Image\n1.0 - Gray]
    Polarity -- Background Light --> Normal[Keep Original Polarity]
    
    %% Enhancement & ROI
    Invert --> Flatten[5. Illumination Correction\nHigh-pass Gaussian Filter subtraction]
    Normal --> Flatten
    Flatten --> CLAHE[6. Local Contrast Enhancement\nCLAHE clip_limit=2.0, grid=8x8]
    
    %% Path Splitting: Segmentation & Denoising
    CLAHE --> Segment[7. ROI Segmentation]
    CLAHE --> Denoise[8. Bilateral Denoising\nEdge-preserving smoothing]
    
    %% Segmentation details
    subgraph Segmentation Engine
        Segment --> Score[Compute ROI Score\n0.72 * local_std + 0.28 * gradient_mag]
        Score --> Otsu[Otsu Automatic Binarization]
        Otsu --> Morph[Morphological Opening/Closing & Fill]
        Morph --> Convex[Convex Hull Optimization]
    end
    
    Convex --> ROIMask[ROI Foreground Mask]
    Denoise --> Orient[9. Orientation & Coherence Map\nStructure Tensor Eigenvalues]
    
    %% Masking Pipeline
    ROIMask --> Masking[10. Parallel Masking Engine]
    Denoise --> Masking
    
    subgraph Masking Engine
        Masking --> Color[Color Saturation Detection\nHSV saturation > 55]
        Masking --> Hough[Straight Rule Lines\nHough Line P Transform]
        Masking --> Adaptive[Text & Writing\nAdaptive Threshold component analysis]
        Masking --> MedianDiff[Handwriting Strokes\nLocal Median Deviation > 92%]
        Masking --> Manual[Manual Override Geometries\nUser rectangles/polygons]
    end
    
    Color --> AggMask[Unified Artifact Mask]
    Hough --> AggMask
    Adaptive --> AggMask
    MedianDiff --> AggMask
    Manual --> AggMask
    
    %% Guarded Reconstruction Decision Tree
    AggMask --> Guard{11. Reconstruction Guard\nFor each component in Mask}
    ROIMask --> Guard
    Orient --> Guard
    
    subgraph Guarded Reconstruction
        Guard -- Area <= 1600px² AND Width/Height <= 48px\nAND Neighbor Support >= 55% --> Reconstruct[Inpaint Telea Method\nReconstruct Ridge Flow]
        Guard -- Limits Exceeded --> Block[Block Reconstruction\nKeep original pixels, flag warning]
    end
    
    Reconstruct --> RebuiltImg[Reconstructed Ridge Image]
    Block --> RebuiltImg
    
    %% Final Enhancement
    RebuiltImg --> Gabor[12. Local-Tuned Gabor Filter\nConvolved at local angle theta]
    Orient --> Gabor
    Gabor --> Blend[13. Coherence-Guided Blending\nBlend Gabor output where coherence > 0.18]
    Blend --> PostDenoise[14. Post-Denoising & Background Suppression\nZero background outside ROI]
    
    %% Output Generation
    PostDenoise --> Output[Final Processed Evidence]
    PostDenoise --> Metrics[15. Quality Metrics Generation]
    
    %% Styles
    classDef engine fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#f8fafc;
    classDef decision fill:#0f172a,stroke:#eab308,stroke-width:2px,color:#f8fafc;
    classDef io fill:#020617,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    class Load,Gray,Stretch,Flatten,CLAHE,Denoise,Orient,Gabor,Blend,PostDenoise,Score,Otsu,Morph,Convex,Color,Hough,Adaptive,MedianDiff,Manual,Reconstruct,Block,Metrics engine;
    class Polarity,Guard decision;
    class Input,Output io;
```

---

## Core Algorithmic Engineering & Mathematical Formulations

This section provides the rigorous mathematical definitions and algorithmic execution details for each step of the pipeline.

### 1. Robust Gray Normalization & Polarity Check
- **Robust Percentile Contrast Rescaling**: To prevent extreme pixel values (dust, scanner glare, sensor clipping) from compressing the dynamic range of fingerprint ridges, we compute the low and high intensity percentiles $I_{p_{\text{low}}}$ (default 1.0%) and $I_{p_{\text{high}}}$ (default 99.0%). The rescaled floating-point intensity $I_{\text{out}}(x,y)$ is defined as:
  $$I_{\text{out}}(x,y) = \text{clamp}\left( \frac{I(x,y) - I_{p_{\text{low}}}}{I_{p_{\text{high}}} - I_{p_{\text{low}}}}, \; 0.0, \; 1.0 \right)$$
- **Automatic Polarity Inversion**: Latent prints can be captured on light backgrounds (dark ridges) or dark backing cards (light ridges). To guarantee a standardized representation (dark ridges on a light background), the median intensity of the outer border boundary region $M_b$ (within a margin of $w/16$) is compared with the median intensity of the central region $M_c$:
  $$I_{\text{polar}}(x,y) = \begin{cases} 
  1.0 - I_{\text{out}}(x,y) & \text{if } M_b < 0.42 \text{ and } M_c > M_b + 0.08 \\ 
  I_{\text{out}}(x,y) & \text{otherwise} 
  \end{cases}$$

### 2. Illumination Correction & CLAHE
- **Illumination Correction via Background Subtraction**: uneven illumination is corrected by modeling the low-frequency background luminance via a large Gaussian blur kernel $G_{\sigma}$ where standard deviation $\sigma = 26.0$. Subtracting this low-frequency baseline flattens the background:
  $$I_{\text{flat}}(x,y) = I_{\text{polar}}(x,y) - \left( I_{\text{polar}} * G_{\sigma} \right)(x,y) + \text{median}\left( I_{\text{polar}} * G_{\sigma} \right)$$
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: To prevent noise amplification while highlighting local ridge detail, the image is divided into $M \times N$ local tiles (default $8 \times 8$). For each tile, the local histogram $H(k)$ is computed. Values above a clip limit $\beta$ (default 2.0) are redistributed uniformly across all bins before mapping via the cumulative distribution function (CDF). Inter-tile boundary artifacts are suppressed via bilinear interpolation:
  $$\beta = \alpha \cdot \frac{N_x \cdot N_y}{L}$$
  where $N_x, N_y$ are the tile dimensions, $L$ is the number of gray levels, and $\alpha$ is the clip factor.

### 3. Texture-Based Fingerprint ROI Segmentation
- **Texture-Gradient Composite Score**: Fingerprint ridges are characterized by structured contrast and directional edges. The foreground score map $S(x,y)$ combines local standard deviation and gradient magnitude:
  $$S(x,y) = w_v \cdot \sigma_{\text{local}}(x,y) + w_g \cdot \sqrt{I_x(x,y)^2 + I_y(x,y)^2}$$
  where $w_v = 0.72$, $w_g = 0.28$, and the local variance is computed over a sliding block $W$ of size $23 \times 23$:
  $$\sigma_{\text{local}}(x,y) = \sqrt{\frac{1}{|W|} \sum_{(u,v) \in W} \left( I(u,v) - \mu(x,y) \right)^2}$$
  $I_x$ and $I_y$ are estimated using Sobel kernels.
- **Morphological Refining & Convex Hull**: The score map is binarized via Otsu's thresholding. The resulting mask $M_{\text{raw}}$ is refined to establish a smooth contiguous boundary:
  $$M_{\text{segmented}} = \text{ConvexHull}\left( \left( M_{\text{raw}} \bullet S_{\text{disk}(9)} \right) \circ S_{\text{disk}(3)} \right)$$
  where $\bullet$ represents morphological closing, $\circ$ represents morphological opening, and the structuring elements $S$ are disks of specified radii.

### 4. Bilateral Denoising & Structure Tensor Parameter Estimation
- **Edge-Preserving Bilateral Filter**: Denoising is performed using a bilateral filter to smooth sensor and paper grain noise without degrading the high-frequency boundaries of ridge edges:
  $$I_{\text{denoised}}(x,y) = \frac{1}{W_p(x,y)} \sum_{(u,v) \in \Omega} I(u,v) \exp\left( -\frac{(u-x)^2 + (v-y)^2}{2\sigma_d^2} \right) \exp\left( -\frac{(I(u,v)-I(x,y))^2}{2\sigma_r^2} \right)$$
  where $\Omega$ is the neighborhood window (diameter $d=5$), $\sigma_d = 5.0$ (spatial variance), $\sigma_r = 35.0$ (range color variance), and $W_p$ is the normalization sum.
- **Structure Tensor Construction**: The local orientation field is estimated by computing the symmetric structure tensor $J$ from smoothed gradients $I_x, I_y$:
  $$J(x,y) = \begin{bmatrix} 
  \langle I_x^2 \rangle_w & \langle I_x I_y \rangle_w \\ 
  \langle I_x I_y \rangle_w & \langle I_y^2 \rangle_w 
  \end{bmatrix}$$
  where $\langle \cdot \rangle_w$ denotes local integration averaging convolved with a Gaussian window $G_{\sigma_i}$ ($\sigma_i = 5.0$).
- **Local Orientation & Coherence**: The dominant ridge direction angle $\theta(x,y) \in [0, \pi)$ is perpendicular to the primary gradient eigenvector:
  $$\theta(x,y) = \frac{1}{2} \arctan2\left( 2\langle I_x I_y \rangle_w, \; \langle I_x^2 \rangle_w - \langle I_y^2 \rangle_w \right) + \frac{\pi}{2}$$
  Coherence $C(x,y)$ measures the local structural anisotropy, indicating the alignment strength of the ridges:
  $$C(x,y) = \frac{\sqrt{ \left(\langle I_x^2 \rangle_w - \langle I_y^2 \rangle_w \right)^2 + 4\langle I_x I_y \rangle_w^2 }}{\langle I_x^2 \rangle_w + \langle I_y^2 \rangle_w + \epsilon}$$
  where $\epsilon = 10^{-6}$ prevents division by zero.

### 5. Multi-Source Masking Engine
- **HSV Ink Saturation Masking**: Writing inks (stamps, pens) differ from grayscale ridges in the color saturation space. Pixels are masked if they exceed saturation and gray-deviation thresholds in the HSV space:
  $$M_{\text{color}}(x,y) = \left( S_{\text{HSV}}(x,y) > 55 \right) \land \left( |V_{\text{HSV}}(x,y) - I_{\text{gray}}(x,y)| \cdot 255 > 30 \right)$$
- **Hough Line Grid Masking**: Straight ruled lines are extracted by mapping Canny edge points to the parametric Hough accumulator space $(\rho, \phi)$:
  $$\rho = x \cos \phi + y \sin \phi$$
  Accumulator bin values exceeding $80$ votes are back-projected to locate line segments. Segments longer than $34\%$ of the image's minimum dimension are masked and dilated by a width of $5\text{ px}$.
- **Adaptive handwriting Stroke Masking**: Thick handwriting strokes are extracted using a local median filter comparison. Pixels showing a large negative intensity deviation from the local median are masked:
  $$\Delta I(x,y) = \text{median}_{23\times23}\left( I_{\text{base}} \right)(x,y) - I_{\text{base}}(x,y)$$
  $$M_{\text{stroke}}(x,y) = \text{MorphologicalClosing}\left( \Delta I(x,y) > P_{92}(\Delta I), \; S_{\text{disk}(2)} \right)$$
  where $P_{92}$ represents the $92\text{nd}$ percentile threshold computed over the segmented ROI region.

### 6. Guarded Reconstruction & Telea Inpainting
- **Guarded Inpainting Decision Rules**: For each connected component $K_i$ in the cumulative artifact mask $M_{\text{artifact}}$, reconstruction is only permitted if the component is small enough and has sufficient surrounding structure support to prevent the creation of false ridge details:
  $$\text{Reconstruct}(K_i) = \begin{cases} 
  \text{True} & \text{if } \text{Area}(K_i) \le 1600\text{ px}^2 \land \max(\text{Width}(K_i), \text{Height}(K_i)) \le 48\text{ px} \land \text{Support}(K_i) \ge 0.55 \\ 
  \text{False} & \text{otherwise} 
  \end{cases}$$
- **Neighborhood Support Ratio**: The support ratio evaluates the density of valid, clean fingerprint ridges surrounding the artifact component boundary:
  $$\text{Support}(K_i) = \frac{\sum_{(u,v) \in \partial K_i} M_{\text{ROI}}(u,v) \cdot \left( 1 - M_{\text{artifact}}(u,v) \right)}{\sum_{(u,v) \in \partial K_i} 1}$$
  where $\partial K_i$ is a dilated ring (width $6\text{ px}$) around the component boundary.
- **Telea Fast Marching Inpainting**: Gaps where reconstruction is allowed are filled using Telea's method, which propagates image values inwards along the normal directions of the boundary interface, solving the Eikonal equation:
  $$\|\nabla T\| = 1$$
  where $T$ represents the time-of-arrival distance field. Gaps where reconstruction is blocked remain unaltered in the final output and are flagged with a quality warning.

### 7. Orientation-Selective Gabor Enhancement
- **Anisotropic Gabor Filtering**: The convolved enhanced image $E(x,y)$ is computed convolving the local neighborhood with a Gabor kernel $g$ tuned to the local ridge orientation angle $\theta(x,y)$ and standard ridge period $\lambda = 9.0\text{ px}$:
  $$g(x,y; \theta, \lambda, \sigma, \gamma) = \exp\left( -\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2} \right) \cos\left( 2\pi\frac{x'}{\lambda} \right)$$
  $$x' = x \cos\theta + y \sin\theta, \quad y' = -x \sin\theta + y \cos\theta$$
  where $\sigma = 4.0$ controls the Gaussian envelope bandwidth and $\gamma = 0.55$ defines the spatial aspect ratio.
- **Coherence Blending**: To prevent introducing artificial patterns in unstructured regions, Gabor output is blended with the denoised image using the computed local coherence map $C(x,y)$ as a weight in reliable ROI regions:
  $$I_{\text{enhanced}}(x,y) = \begin{cases} 
  (1 - w_b) \cdot I(x,y) + w_b \cdot \left(I * g_{\theta}\right)(x,y) & \text{if } (x,y) \in M_{\text{ROI}} \land C(x,y) > 0.18 \\ 
  I(x,y) & \text{otherwise} 
  \end{cases}$$
  where $w_b = 0.32$ is the maximum blending weight.

---

## Workspace Layout

```
.
├── fingerprint_pipeline.py  # Core algorithms and image processing logic
├── web_app.py               # Asynchronous Flask server and glassmorphic UI
├── requirements.txt         # Package dependencies
├── run_ui.ps1               # PowerShell launcher for the web console
├── run_notebook.ps1         # PowerShell launcher for Jupyter
├── scripts/
│   └── build_notebook.py    # Utility script to compile notebooks
├── notebooks/
│   └── Fingerprint_Forensic_Preprocessing_Reconstruction.ipynb
├── utils/                   # Test datasets and sample images
└── outputs/                 # Output directory for processed cases (git-ignored)
```

---

## Git Workflow Guide for Contributors

To maintain a clean, stable history, follow this workflow:

### 1. Main Branch Policy
- The `main` branch represents fully verified, build-passing, and production-tested releases. 
- Direct pushes to `main` are restricted. All contributions must arrive via Pull Requests.

### 2. Feature Development
Create a descriptively named branch for your work:
```bash
git checkout -b feature/adaptive-denoising-optimization
```

### 3. Clean Commits & Semantic Formatting
Structure your commit messages using semantic prefixes to help automated change tracking:
- `feat:` for new capabilities (e.g. `feat: add adaptive saturation thresholding`)
- `fix:` for bugs (e.g. `fix: resolve structure tensor boundary scale check`)
- `docs:` for documentation (e.g. `docs: update math equations in README`)
- `refactor:` for code cleanups that do not change logic.

### 4. Syncing & Merging
To prevent merge conflict commits, rebase your feature branch on `main` before merging:
```bash
git checkout main
git pull origin main
git checkout feature/your-feature
git rebase main
```

### 5. Pushing & Authentication
If pushing from the terminal for the first time, authenticate via the HTTPS URL:
```bash
git push -u origin feature/your-feature
```
This triggers the Git Credential Manager GUI to secure your session.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kushalkhadkaa/Biometrics.git
   cd Biometrics
   ```
2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Install standard dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
