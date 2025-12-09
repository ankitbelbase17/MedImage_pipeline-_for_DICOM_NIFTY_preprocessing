import os
import numpy as np
import pydicom
from scipy import ndimage
import nibabel as nib
from typing import List, Tuple, Optional, Union


class DICOMProcessor:
    """
    A comprehensive DICOM processing class for medical imaging.
    
    Features:
    - Load and sort DICOM series
    - Convert to Hounsfield Units (HU)
    - Resample to isotropic spacing
    - Apply windowing for different tissue types
    - Save as NIfTI format
    """
    
    # Predefined window presets
    WINDOW_PRESETS = {
        'lung': {'min': -1000, 'max': 400},
        'soft_tissue': {'min': -160, 'max': 240},
        'bone': {'min': 400, 'max': 1800},
        'brain': {'min': 0, 'max': 80},
        'liver': {'min': -150, 'max': 250},
        'mediastinum': {'min': -175, 'max': 275},
    }
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DICOM processor.
        
        Args:
            verbose: Print processing information
        """
        self.verbose = verbose
        self.slices = None
        self.pixel_spacing = None
        self.slice_thickness = None
        
    def load_scan(self, path: str) -> List[pydicom.FileDataset]:
        """
        Load and sort DICOM series from a directory.
        
        Args:
            path: Directory containing DICOM files
            
        Returns:
            List of sorted DICOM slices
            
        Example:
            >>> processor = DICOMProcessor()
            >>> slices = processor.load_scan('./patient_ct/')
            >>> print(f"Loaded {len(slices)} slices")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        # Load all DICOM files
        dicom_files = [f for f in os.listdir(path) if not f.startswith('.')]
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {path}")
        
        if self.verbose:
            print(f"Loading {len(dicom_files)} DICOM files...")
        
        slices = []
        for filename in dicom_files:
            try:
                ds = pydicom.dcmread(os.path.join(path, filename))
                slices.append(ds)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {filename}: {e}")
                continue
        
        if not slices:
            raise ValueError("No valid DICOM files could be loaded")
        
        # Sort by ImagePositionPatient (Z-axis)
        # This ensures slices are in correct anatomical order
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            if self.verbose:
                print("Warning: ImagePositionPatient not found, sorting by InstanceNumber")
            slices.sort(key=lambda x: int(x.InstanceNumber))
        
        self.slices = slices
        
        # Extract spacing information
        self._extract_spacing()
        
        if self.verbose:
            print(f"✓ Loaded {len(slices)} slices")
            print(f"  Patient: {slices[0].PatientName}")
            print(f"  Study: {slices[0].StudyDescription if hasattr(slices[0], 'StudyDescription') else 'N/A'}")
            print(f"  Modality: {slices[0].Modality}")
            print(f"  Original spacing: {self.pixel_spacing[0]:.2f} × {self.pixel_spacing[1]:.2f} × {self.slice_thickness:.2f} mm")
        
        return slices
    