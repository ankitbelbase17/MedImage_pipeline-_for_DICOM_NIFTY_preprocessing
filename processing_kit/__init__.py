
"""
MedImagePro - Medical Image Processing Toolkit
================================================

A comprehensive Python toolkit for medical image processing,
supporting DICOM and NIfTI formats.

Main Classes:
    - DICOMProcessor: DICOM file processing
    - NIfTIProcessor: NIfTI file processing
    - MeshConverter: 3D mesh operations
    - Pipeline: Complete processing workflow

Quick Start:
    >>> from medimagekit import DICOMProcessor, NIfTIProcessor
    >>> 
    >>> # Process DICOM
    >>> dicom_proc = DICOMProcessor()
    >>> volume = dicom_proc.load_scan('./dicom_folder/')
    >>> 
    >>> # Process NIfTI
    >>> nifti_proc = NIfTIProcessor()
    >>> data, affine, header = nifti_proc.load('brain.nii.gz')

Author: Your Name
License: MIT
Version: 0.1.0
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__license__ = 'MIT'

# Core processors
from .dicom_processor import DICOMProcessor, process_dicom_folder
from .nifti_processor import NIfTIProcessor, load_nifti

# Import other modules (when they exist)
try:
    from .mesh_converter import MeshConverter
except ImportError:
    MeshConverter = None

try:
    from .visualization import Visualizer
except ImportError:
    Visualizer = None

try:
    from .filters import Filters
except ImportError:
    Filters = None

try:
    from .transformations import Transformations
except ImportError:
    Transformations = None

# Main exports
__all__ = [
    'DICOMProcessor',
    'NIfTIProcessor',
    'MeshConverter',
    'Visualizer',
    'Filters',
    'Transformations',
    'process_dicom_folder',
    'load_nifti',
    '__version__',
]

# Print welcome message on import
def _show_welcome():
    """Display welcome message."""
    print(f"MedImagePro v{__version__}")
    print("Medical Image Processing Toolkit")
    print("-" * 40)