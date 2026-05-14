# Partially-Connected-Unrolled-RL-Network (PC-RLN)
Feng Tian and Weijian Yang, "Unrolled Richardson-Lucy deconvolution network with partially connected layers in computational microscopy," Opt. Express 34, 18676-18689 (2026)
### Clone this repository:
```
git clone https://github.com/Yang-Research-Laboratory/Partially-Connected-Unrolled-RL-Network
```

## [Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-10-18676###)
![schematicimage](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure1.png)<br />

## Installation & Requirements

### Prerequisites
- **Python 3.8+** (recommended 3.9 or 3.10)
- **MATLAB R2020b** or later (for index building in FLFM only)
- **Git**

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow>=2.10 numpy scipy scikit-image matplotlib h5py
```

**Key dependencies:**
- **TensorFlow 2.10+** — Deep learning framework for model training and inference
- **NumPy/SciPy** — Numerical computation and MATLAB `.mat` file handling
- **scikit-image** — Image processing utilities
- **Matplotlib** — Visualization and result plotting
- **h5py** — HDF5 file support for data handling

### GPU Setup (Recommended)

Training is significantly faster with GPU support. According to the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-10-18676###), training was performed on **NVIDIA A10G GPUs**:
- **FLFM training**: ~3 GB GPU RAM required
- **MLA training**: ~24 GB GPU RAM recommended (or use smaller batch sizes on smaller GPUs)

**Install TensorFlow with GPU support:**

```bash
# For CUDA 11.8 support (recommended)
pip install tensorflow[and-cuda]>=2.10

# Or specify manually:
pip install tensorflow>=2.10 tensorflow-io-gcs-filesystem
```

Verify GPU setup:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Should show available GPUs
```

### Clone the Repository

```bash
git clone https://github.com/Yang-Research-Laboratory/Partially-Connected-Unrolled-RL-Network
cd Partially-Connected-Unrolled-RL-Network
```

## Quick Start

### For Fourier Light Field Microscopy (FLFM)

1. **Prepare the index mapping** (MATLAB, optional if `data_FLFM_index.mat` exists):
   ```matlab
   cd Fourier\ Light\ Field\ Microscopy/
   buildindex_FLFM
   ```

2. **Train and test the PC-RLN model** (Python):
   ```bash
   cd Fourier\ Light\ Field\ Microscopy/
   python PCRLNet_FLFM.py
   ```

### For Microlens Array Microscopy (MLA)

Run the training directly (pre-computed index mapping provided):

```bash
cd Microlens\ Array\ Microscopy/
python PCRLNet_MLA.py
```

---

## Fourier Light Field Microscopy (FLFM) — Detailed Guide
<p align="center">
  <img src="https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure2.png" width="80%">
</p>

### Dataset

**Provided files:**
- [**PSF_FLFM.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/PSF_FLFM.mat) — 3D point spread function
- [**data_FLFM_th08.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/data_FLFM.mat) — Simulated FLFM dataset (60 training, 8 testing samples)
- Dataset adopted from Yi et al. ([**F-VCD**](https://www.nature.com/articles/s42003-023-05636-x)) [1]

### Step 1: Generate Index Mapping (MATLAB, optional)

If [**data_FLFM_index.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/data_FLFM_index.mat) doesn't exist, extract pixel-voxel mapping using MATLAB:

```matlab
cd Fourier\ Light\ Field\ Microscopy/
buildindex_FLFM
```

This generates voxel-pixel coordinates and initializes weights from PSF intensity for partially connected layers.

### Step 2: Train & Test PC-RLN (Python)

Construct the PC-RLN architecture and train on the dataset:

```bash
cd Fourier\ Light\ Field\ Microscopy/
python PCRLNet_FLFM.py
```

## Microlens Array Microscopy (MLA) — Detailed Guide
<p align="center">
  <img src="https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure3.png" width="70%">
</p>

### Dataset

**Provided files:**
- [**data_MLA.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Microlens%20Array%20Microscopy/data_MLA.mat) — Simulated dataset with randomly positioned alphabet letters
- [**data_MLA_index.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Microlens%20Array%20Microscopy/data_MLA_index.mat) — Pre-computed pixel-voxel indexing map
- **System:** Microlens array (MLA) imaging with spatially varying PSF, 13 depth levels

### Train & Test PC-RLN (Python)

Construct the PC-RLN architecture and train on the dataset:

```bash
cd Microlens\ Array\ Microscopy/
python PCRLNet_MLA.py
```




[1]. Yi, C., Zhu, L., Sun, J. et al. Video-rate 3D imaging of living cells using Fourier view-channel-depth light field microscopy. Commun Biol 6, 1259 (2023). https://doi.org/10.1038/s42003-023-05636-x
