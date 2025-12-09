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
        
        Formula: HU = pixel_value x slope + intercept
        
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
    
    def resample(
        self, 
        image: np.ndarray, 
        scan: Optional[List[pydicom.FileDataset]] = None,
        new_spacing: List[float] = [1.0, 1.0, 1.0],
        order: int = 1
    ) -> np.ndarray:
        """
        Resample image to isotropic spacing.
        
        This is crucial for accurate 3D reconstruction and measurements.
        
        Args:
            image: 3D numpy array
            scan: DICOM slices (uses self.slices if None)
            new_spacing: Target spacing in mm [Z, X, Y]
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            
        Returns:
            Resampled 3D numpy array
            
        Example:
            >>> # Resample to 1mm isotropic
            >>> resampled = processor.resample(hu_array, new_spacing=[1, 1, 1])
            >>> # Resample to 0.5mm (higher resolution)
            >>> high_res = processor.resample(hu_array, new_spacing=[0.5, 0.5, 0.5])
        """
        if scan is None:
            scan = self.slices
        
        if scan is None:
            raise ValueError("No DICOM slices available. Provide scan parameter.")
        
        if self.verbose:
            print(f"Resampling to {new_spacing} mm spacing...")
        
        # Get current spacing
        spacing = np.array([
            self.slice_thickness,
            self.pixel_spacing[0],
            self.pixel_spacing[1]
        ], dtype=np.float32)
        
        # Calculate resize factor
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        
        if self.verbose:
            print(f"  Original shape: {image.shape}")
            print(f"  Target shape: {tuple(new_shape.astype(int))}")
            print(f"  Resize factor: {real_resize_factor}")
        
        # Perform resampling using spline interpolation
        new_image = ndimage.zoom(image, real_resize_factor, order=order)
        
        if self.verbose:
            print(f"✓ Resampled. New shape: {new_image.shape}")
        
        return new_image
    
    def apply_window(
        self, 
        image: np.ndarray, 
        window: Union[str, dict] = 'soft_tissue'
    ) -> np.ndarray:
        """
        Apply windowing (contrast adjustment) for specific tissue visualization.
        
        Args:
            image: 3D array in Hounsfield Units
            window: Either preset name or dict with 'min' and 'max' keys
            
        Returns:
            Windowed and normalized array [0, 1]
            
        Available presets:
            - 'lung': [-1000, 400]
            - 'soft_tissue': [-160, 240]
            - 'bone': [400, 1800]
            - 'brain': [0, 80]
            - 'liver': [-150, 250]
            - 'mediastinum': [-175, 275]
            
        Example:
            >>> # Use preset
            >>> lung_view = processor.apply_window(hu_array, 'lung')
            >>> # Custom window
            >>> custom_view = processor.apply_window(hu_array, {'min': -500, 'max': 500})
        """
        if isinstance(window, str):
            if window not in self.WINDOW_PRESETS:
                raise ValueError(
                    f"Unknown window preset: {window}. "
                    f"Available: {list(self.WINDOW_PRESETS.keys())}"
                )
            window_params = self.WINDOW_PRESETS[window]
        else:
            window_params = window
        
        min_hu = window_params['min']
        max_hu = window_params['max']
        
        if self.verbose:
            print(f"Applying window: [{min_hu}, {max_hu}] HU")
        
        # Clip values
        windowed = np.clip(image, min_hu, max_hu)
        
        # Normalize to [0, 1]
        windowed = (windowed - min_hu) / (max_hu - min_hu)
        
        if self.verbose:
            print(f"✓ Applied windowing. Range: [{windowed.min():.3f}, {windowed.max():.3f}]")
        
        return windowed
    
    def normalize_and_clip(
        self, 
        image: np.ndarray, 
        min_hu: float = -1000, 
        max_hu: float = 400
    ) -> np.ndarray:
        """
        Clip HU range and normalize to [0, 1].
        
        Args:
            image: 3D array in Hounsfield Units
            min_hu: Minimum HU value to keep
            max_hu: Maximum HU value to keep
            
        Returns:
            Clipped and normalized array
            
        Example:
            >>> normalized = processor.normalize_and_clip(hu_array, 
            ...                                           min_hu=-1000, 
            ...                                           max_hu=400)
        """
        return self.apply_window(image, {'min': min_hu, 'max': max_hu})
    
    def save_as_nifti(
        self, 
        image: np.ndarray, 
        output_path: str,
        affine: Optional[np.ndarray] = None
    ):
        """
        Save processed volume as NIfTI file.
        
        Args:
            image: 3D numpy array
            output_path: Output file path (.nii or .nii.gz)
            affine: 4×4 affine matrix (creates identity if None)
            
        Example:
            >>> processor.save_as_nifti(processed_volume, 'output.nii.gz')
        """
        if affine is None:
            # Create identity affine with proper spacing
            affine = np.eye(4)
            if self.pixel_spacing is not None:
                affine[0, 0] = self.pixel_spacing[0]
                affine[1, 1] = self.pixel_spacing[1]
                affine[2, 2] = self.slice_thickness
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(image, affine)
        
        # Save
        nib.save(nifti_img, output_path)
        
        if self.verbose:
            print(f"✓ Saved NIfTI: {output_path}")
    
    def get_metadata(self) -> dict:
        """
        Extract metadata from loaded DICOM series.
        
        Returns:
            Dictionary with DICOM metadata
            
        Example:
            >>> metadata = processor.get_metadata()
            >>> print(metadata['patient_name'])
        """
        if not self.slices:
            raise ValueError("No DICOM slices loaded")
        
        ds = self.slices[0]
        
        metadata = {
            'patient_name': str(ds.PatientName) if hasattr(ds, 'PatientName') else 'Unknown',
            'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else 'Unknown',
            'study_date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else 'Unknown',
            'modality': str(ds.Modality),
            'manufacturer': str(ds.Manufacturer) if hasattr(ds, 'Manufacturer') else 'Unknown',
            'num_slices': len(self.slices),
            'pixel_spacing': self.pixel_spacing.tolist() if self.pixel_spacing is not None else None,
            'slice_thickness': float(self.slice_thickness) if self.slice_thickness else None,
            'image_shape': self.slices[0].pixel_array.shape,
        }
        
        return metadata
    
    def process_folder(
        self, 
        input_path: str, 
        output_path: str,
        new_spacing: List[float] = [1.0, 1.0, 1.0],
        window: str = 'soft_tissue',
        save_format: str = 'nifti'
    ) -> dict:
        """
        Complete pipeline: load, process, and save DICOM series.
        
        Args:
            input_path: Input DICOM folder
            output_path: Output file path
            new_spacing: Target spacing in mm
            window: Window preset name
            save_format: 'nifti' or 'numpy'
            
        Returns:
            Dictionary with processing results
            
        Example:
            >>> result = processor.process_folder(
            ...     './patient_001/',
            ...     './output/patient_001.nii.gz',
            ...     new_spacing=[1, 1, 1],
            ...     window='lung'
            ... )
        """
        # Load
        slices = self.load_scan(input_path)
        
        # Convert to HU
        hu_array = self.get_pixels_hu(slices)
        
        # Resample
        resampled = self.resample(hu_array, slices, new_spacing=new_spacing)
        
        # Apply windowing
        processed = self.apply_window(resampled, window=window)
        
        # Save
        if save_format == 'nifti':
            self.save_as_nifti(processed, output_path)
        elif save_format == 'numpy':
            np.save(output_path, processed)
            if self.verbose:
                print(f"✓ Saved NumPy: {output_path}")
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        return {
            'original_shape': hu_array.shape,
            'processed_shape': processed.shape,
            'output_path': output_path,
            'metadata': self.get_metadata()
        }


# Convenience function for quick processing
def process_dicom_folder(
    input_path: str,
    output_path: str,
    new_spacing: List[float] = [1.0, 1.0, 1.0],
    window: str = 'soft_tissue',
    verbose: bool = True
) -> np.ndarray:
    """
    Quick DICOM processing with default settings.
    
    Args:
        input_path: DICOM folder path
        output_path: Output NIfTI path
        new_spacing: Target spacing [Z, X, Y] in mm
        window: Window preset
        verbose: Print progress
        
    Returns:
        Processed 3D numpy array
        
    Example:
        >>> from medimagekit.dicom_processor import process_dicom_folder
        >>> volume = process_dicom_folder('./ct_scan/', './output.nii.gz')
    """
    processor = DICOMProcessor(verbose=verbose)
    processor.process_folder(input_path, output_path, new_spacing, window)
    
    # Return processed array
    slices = processor.slices
    hu = processor.get_pixels_hu(slices)
    resampled = processor.resample(hu, slices, new_spacing)
    return processor.apply_window(resampled, window)
