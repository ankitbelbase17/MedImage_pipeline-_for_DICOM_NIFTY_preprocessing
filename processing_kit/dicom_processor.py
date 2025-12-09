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
    
    def _extract_spacing(self):
        """Extract pixel spacing and slice thickness from DICOM metadata."""
        if not self.slices:
            return
        
        # Get pixel spacing (X, Y)
        self.pixel_spacing = np.array(self.slices[0].PixelSpacing, dtype=np.float32)
        
        # Calculate slice thickness (Z)
        if len(self.slices) > 1:
            self.slice_thickness = np.abs(
                self.slices[1].ImagePositionPatient[2] - 
                self.slices[0].ImagePositionPatient[2]
            )
        else:
            self.slice_thickness = self.slices[0].SliceThickness
    
    def get_pixels_hu(self, slices: Optional[List[pydicom.FileDataset]] = None) -> np.ndarray:
        """
        Convert DICOM pixel values to Hounsfield Units (HU).
        
        Formula: HU = pixel_value × slope + intercept
        
        Args:
            slices: List of DICOM slices (uses self.slices if None)
            
        Returns:
            3D numpy array in Hounsfield Units
            
        Example:
            >>> hu_array = processor.get_pixels_hu()
            >>> print(f"HU range: {hu_array.min()} to {hu_array.max()}")
        """
        if slices is None:
            slices = self.slices
        
        if slices is None:
            raise ValueError("No DICOM slices loaded. Call load_scan() first.")
        
        if self.verbose:
            print("Converting to Hounsfield Units...")
        
        # Stack all slices into 3D array
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        
        # Fix common issue: air stored as -2000
        image[image == -2000] = 0
        
        # Get rescale parameters from DICOM header
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        
        if self.verbose:
            print(f"  Rescale slope: {slope}")
            print(f"  Rescale intercept: {intercept}")
        
        # Apply rescale formula
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        
        image += np.int16(intercept)
        
        if self.verbose:
            print(f"✓ Converted to HU. Range: [{image.min()}, {image.max()}]")
        
        return np.array(image, dtype=np.int16)
    