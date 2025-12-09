"""
NIfTI Processing Module
Handles loading, processing, visualization, and transformation of NIfTI medical imaging data.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fftn, ifftn, fftshift
from skimage import filters, measure
from typing import Tuple, Optional, Union, List
import warnings


class NIfTIProcessor:
    """
    Comprehensive NIfTI file processing class.
    
    Features:
    - Load/save NIfTI files
    - Preprocessing pipeline
    - Image transformations
    - Frequency domain operations
    - Visualization
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize NIfTI processor.
        
        Args:
            verbose: Print processing information
        """
        self.verbose = verbose
        self.data = None
        self.affine = None
        self.header = None
    
    def load(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
        """
        Load NIfTI file.
        
        Args:
            filepath: Path to .nii or .nii.gz file
            
        Returns:
            Tuple of (data, affine, header)
            
        Example:
            >>> processor = NIfTIProcessor()
            >>> data, affine, header = processor.load('brain.nii.gz')
        """
        if self.verbose:
            print(f"Loading NIfTI: {filepath}")
        
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        self.data = data
        self.affine = affine
        self.header = header
        
        if self.verbose:
            print(f"✓ Loaded shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")
            if len(data.shape) >= 3:
                voxel_size = header.get_zooms()[:3]
                print(f"  Voxel size: {voxel_size[0]:.2f} × {voxel_size[1]:.2f} × {voxel_size[2]:.2f} mm")
        
        return data, affine, header
    
    def save(self, data: np.ndarray, affine: np.ndarray, filepath: str):
        """
        Save data as NIfTI file.
        
        Args:
            data: 3D/4D numpy array
            affine: 4×4 affine transformation matrix
            filepath: Output path (.nii or .nii.gz)
            
        Example:
            >>> processor.save(processed_data, affine, 'output.nii.gz')
        """
        img = nib.Nifti1Image(data, affine)
        nib.save(img, filepath)
        
        if self.verbose:
            print(f"✓ Saved: {filepath}")
    
    # ========== Preprocessing Methods ==========
    
    def normalize(
        self, 
        data: np.ndarray, 
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normalize data using different methods.
        
        Args:
            data: Input array
            method: 'minmax', 'zscore', or 'percentile'
            
        Returns:
            Normalized array
            
        Example:
            >>> normalized = processor.normalize(data, method='percentile')
        """
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        elif method == 'zscore':
            return (data - data.mean()) / (data.std() + 1e-8)
        
        elif method == 'percentile':
            p1, p99 = np.percentile(data, [1, 99])
            return np.clip((data - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denoise(
        self, 
        data: np.ndarray, 
        method: str = 'gaussian', 
        **kwargs
    ) -> np.ndarray:
        """
        Denoise 3D data.
        
        Args:
            data: Input array
            method: 'gaussian', 'median', or 'bilateral'
            **kwargs: Method-specific parameters
            
        Returns:
            Denoised array
            
        Examples:
            >>> # Gaussian denoising
            >>> denoised = processor.denoise(data, 'gaussian', sigma=1.0)
            >>> # Bilateral denoising (edge-preserving)
            >>> denoised = processor.denoise(data, 'bilateral', 
            ...                               sigma_spatial=2.0, sigma_color=0.05)
        """
        if self.verbose:
            print(f"Denoising with {method} filter...")
        
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            result = ndimage.gaussian_filter(data, sigma=sigma)
        
        elif method == 'median':
            size = kwargs.get('size', 3)
            result = ndimage.median_filter(data, size=size)
        
        elif method == 'bilateral':
            # Note: bilateral filter can be slow on 3D data
            sigma_spatial = kwargs.get('sigma_spatial', 2.0)
            sigma_color = kwargs.get('sigma_color', 0.05)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = filters.denoise_bilateral(
                    data,
                    sigma_spatial=sigma_spatial,
                    sigma_color=sigma_color
                )
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        if self.verbose:
            print(f"✓ Denoised")
        
        return result
    
    def resample(
        self, 
        data: np.ndarray, 
        zoom_factors: Union[float, Tuple[float, ...]], 
        order: int = 3
    ) -> np.ndarray:
        """
        Resample data to different resolution.
        
        Args:
            data: Input array
            zoom_factors: Scale factor(s)
            order: Interpolation order (0-5)
            
        Returns:
            Resampled array
            
        Example:
            >>> # Downsample to half resolution
            >>> downsampled = processor.resample(data, 0.5)
            >>> # Anisotropic resampling
            >>> resampled = processor.resample(data, (1.0, 1.0, 0.5))
        """
        if self.verbose:
            print(f"Resampling with factor {zoom_factors}...")
        
        result = ndimage.zoom(data, zoom_factors, order=order)
        
        if self.verbose:
            print(f"✓ Resampled: {data.shape} → {result.shape}")
        
        return result
    
    def crop(self, data: np.ndarray, margin: int = 10) -> np.ndarray:
        """
        Crop data to remove empty space.
        
        Args:
            data: Input array
            margin: Margin to keep around content (voxels)
            
        Returns:
            Cropped array
            
        Example:
            >>> cropped = processor.crop(data, margin=5)
        """
        if self.verbose:
            print("Cropping...")
        
        # Create binary mask
        threshold = data.mean() * 0.1
        mask = data > threshold
        
        # Find bounding box
        coords = np.array(np.where(mask))
        
        if coords.size == 0:
            if self.verbose:
                print("Warning: No content found, returning original")
            return data
        
        x_min, y_min, z_min = coords.min(axis=1) - margin
        x_max, y_max, z_max = coords.max(axis=1) + margin
        
        # Ensure within bounds
        x_min, y_min, z_min = max(0, x_min), max(0, y_min), max(0, z_min)
        x_max = min(data.shape[0], x_max)
        y_max = min(data.shape[1], y_max)
        z_max = min(data.shape[2], z_max)
        
        result = data[x_min:x_max, y_min:y_max, z_min:z_max]
        
        if self.verbose:
            print(f"✓ Cropped: {data.shape} → {result.shape}")
        
        return result
    
    def preprocess_pipeline(
        self, 
        data: np.ndarray,
        denoise: bool = True,
        normalize: bool = True,
        crop: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input array
            denoise: Apply denoising
            normalize: Apply normalization
            crop: Apply cropping
            **kwargs: Additional parameters for individual steps
            
        Returns:
            Preprocessed array
            
        Example:
            >>> processed = processor.preprocess_pipeline(data,
            ...                                           denoise=True,
            ...                                           normalize=True,
            ...                                           crop=True)
        """
        if self.verbose:
            print("=" * 50)
            print("PREPROCESSING PIPELINE")
            print("=" * 50)
        
        result = data.copy()
        
        if denoise:
            denoise_method = kwargs.get('denoise_method', 'gaussian')
            denoise_params = kwargs.get('denoise_params', {'sigma': 1.0})
            result = self.denoise(result, denoise_method, **denoise_params)
        
        if normalize:
            normalize_method = kwargs.get('normalize_method', 'percentile')
            result = self.normalize(result, normalize_method)
        
        if crop:
            crop_margin = kwargs.get('crop_margin', 5)
            result = self.crop(result, margin=crop_margin)
        
        if self.verbose:
            print("=" * 50)
            print(f"✓ Pipeline complete: {data.shape} → {result.shape}")
            print("=" * 50)
        
        return result
    
    # ========== Transformation Methods ==========
    
    def rotate_3d(
        self, 
        data: np.ndarray, 
        angle: float, 
        axis: int = 2, 
        reshape: bool = False
    ) -> np.ndarray:
        """
        Rotate 3D data around specified axis.
        
        Args:
            data: Input array
            angle: Rotation angle in degrees
            axis: Rotation axis (0=X, 1=Y, 2=Z)
            reshape: Expand array to fit rotated data
            
        Returns:
            Rotated array
            
        Example:
            >>> rotated = processor.rotate_3d(data, angle=15, axis=2)
        """
        axes_map = [(1, 2), (0, 2), (0, 1)]
        axes = axes_map[axis]
        
        return ndimage.rotate(data, angle, axes=axes, reshape=reshape, order=1)
    
    def flip(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Flip data along specified axis.
        
        Args:
            data: Input array
            axis: Axis to flip (0=X, 1=Y, 2=Z)
            
        Returns:
            Flipped array
        """
        return np.flip(data, axis=axis)
    
    def elastic_deformation(
        self, 
        data: np.ndarray, 
        alpha: float = 10, 
        sigma: float = 3
    ) -> np.ndarray:
        """
        Apply elastic deformation for data augmentation.
        
        Args:
            data: Input array
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            
        Returns:
            Deformed array
            
        Example:
            >>> deformed = processor.elastic_deformation(data, alpha=10, sigma=3)
        """
        shape = data.shape
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        dz = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        
        # Create coordinate grid
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply displacement
        indices = (x + dx, y + dy, z + dz)
        
        return ndimage.map_coordinates(data, indices, order=1, mode='reflect')
    
    # ========== Frequency Domain Methods ==========
    
    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D FFT.
        
        Args:
            data: Input array
            
        Returns:
            Tuple of (fft_data, magnitude, phase)
            
        Example:
            >>> fft_data, magnitude, phase = processor.compute_fft(data)
        """
        fft_data = fftshift(fftn(data))
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        return fft_data, magnitude, phase
    
    def inverse_fft(self, fft_data: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT.
        
        Args:
            fft_data: FFT data
            
        Returns:
            Real-valued spatial domain data
        """
        return np.real(ifftn(fftshift(fft_data)))
    
    def frequency_filter(
        self, 
        data: np.ndarray, 
        filter_type: str = 'lowpass', 
        cutoff: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply frequency domain filter.
        
        Args:
            data: Input array
            filter_type: 'lowpass', 'highpass', or 'bandpass'
            cutoff: Cutoff frequency (0-1)
            
        Returns:
            Tuple of (filtered_data, filter_mask)
            
        Example:
            >>> # Remove high frequencies (smoothing)
            >>> filtered, mask = processor.frequency_filter(data, 'lowpass', 0.3)
            >>> # Extract edges
            >>> edges, mask = processor.frequency_filter(data, 'highpass', 0.1)
        """
        if self.verbose:
            print(f"Applying {filter_type} filter (cutoff={cutoff})...")
        
        # Compute FFT
        fft_data, magnitude, phase = self.compute_fft(data)
        
        # Create frequency mask
        shape = data.shape
        center = [s // 2 for s in shape]
        
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        distance = np.sqrt(
            (x - center[0])**2 + 
            (y - center[1])**2 + 
            (z - center[2])**2
        )
        max_distance = np.sqrt(sum([s**2 for s in center]))
        normalized_distance = distance / max_distance
        
        # Create mask based on filter type
        if filter_type == 'lowpass':
            mask = normalized_distance < cutoff
        elif filter_type == 'highpass':
            mask = normalized_distance > cutoff
        elif filter_type == 'bandpass':
            mask = (normalized_distance > cutoff * 0.5) & (normalized_distance < cutoff)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter
        filtered_fft = fft_data * mask
        filtered_data = self.inverse_fft(filtered_fft)
        
        if self.verbose:
            print(f"✓ Filtered")
        
        return filtered_data, mask
    
    # ========== Visualization Methods ==========
    
    def visualize(
        self, 
        data: np.ndarray, 
        slice_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize NIfTI data with multiple views.
        
        Args:
            data: 3D array to visualize
            slice_idx: Slice index (uses middle if None)
            figsize: Figure size
            
        Example:
            >>> processor.visualize(data, slice_idx=90)
        """
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Axial
        axes[0, 0].imshow(data[:, :, slice_idx], cmap='gray')
        axes[0, 0].set_title(f'Axial (Z={slice_idx})')
        axes[0, 0].axis('off')
        
        # Sagittal
        axes[0, 1].imshow(data[data.shape[0]//2, :, :], cmap='gray')
        axes[0, 1].set_title('Sagittal (X=mid)')
        axes[0, 1].axis('off')
        
        # Coronal
        axes[0, 2].imshow(data[:, data.shape[1]//2, :], cmap='gray')
        axes[0, 2].set_title('Coronal (Y=mid)')
        axes[0, 2].axis('off')
        
        # Histogram
        axes[1, 0].hist(data.flatten(), bins=100, alpha=0.7)
        axes[1, 0].set_title('Intensity Histogram')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        # MIP
        mip = np.max(data, axis=2)
        axes[1, 1].imshow(mip, cmap='gray')
        axes[1, 1].set_title('Maximum Intensity Projection')
        axes[1, 1].axis('off')
        
        # Statistics
        stats_text = f"""
        Shape: {data.shape}
        Mean: {data.mean():.2f}
        Std: {data.std():.2f}
        Min: {data.min():.2f}
        Max: {data.max():.2f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


# Convenience function
def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Quick load of NIfTI file.
    
    Example:
        >>> from medimagekit.nifti_processor import load_nifti
        >>> data, affine, header = load_nifti('brain.nii.gz')
    """
    processor = NIfTIProcessor(verbose=False)
    return processor.load(filepath)