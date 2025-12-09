# MedImage_pipeline-_for_DICOM_NIFTY_preprocessing
# MedImagePro 🏥🧠

> A comprehensive Python toolkit for medical image processing, supporting DICOM and NIfTI formats with advanced preprocessing, visualization, and 3D mesh conversion capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### 🔬 DICOM Processing
- Load and sort DICOM series automatically
- Convert pixel values to Hounsfield Units (HU)
- Resample to isotropic spacing (1×1×1 mm)
- Window-level adjustments for different tissue types
- Batch processing support

### 🧠 NIfTI Processing
- Read/write NIfTI-1 and NIfTI-2 formats
- Multi-planar visualization (axial, sagittal, coronal)
- Advanced preprocessing pipeline
- Affine transformation handling
- 4D fMRI support

### 🎨 Image Processing
- **Filtering**: Gaussian, median, bilateral
- **Transformations**: Rotation, flipping, elastic deformation
- **Frequency domain**: FFT, lowpass/highpass/bandpass filters
- **Normalization**: Min-max, z-score, percentile-based

### 🎭 3D Mesh Operations
- NIfTI/DICOM → 3D mesh conversion (marching cubes)
- Mesh smoothing and optimization
- Mesh → voxel reconstruction
- Export to STL, OBJ, PLY formats

### 📊 Visualization
- Interactive 3D visualizations
- Maximum Intensity Projection (MIP)
- Side-by-side comparisons
- Frequency domain analysis
