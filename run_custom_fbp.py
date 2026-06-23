# ==========================================================================
# Modified from original helix2fan project under Apache License Version 2.0
# Modifications include:
# - Chronological DICOM sorting
# - High-performance vectorized rebinning 
# - Smart ASTRA/Numba FBP reconstruction pipeline (removed torch_radon)
# ==========================================================================

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numba
import tqdm
import sys
import time

# Ensure helix2fan is in the path to import the helper module
sys.path.append('/home/BArch/.gemini/antigravity/scratch/helix2fan')
try:
    from helper import load_tiff_stack_with_metadata, save_to_tiff_stack
except ImportError:
    print("Error: Could not import helper from helix2fan.")
    exit(1)


@numba.njit(parallel=True, fastmath=True)
def fan_beam_backprojection_ultra(filtered_sino, angles, image_size, dso, dsd, du, det_count):
    reco = np.zeros((image_size, image_size), dtype=np.float32)
    
    dtheta = angles[1] - angles[0] if len(angles) > 1 else 1.0
    num_angles = len(angles)
    
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    
    u_offset = det_count / 2.0 - 0.5
    dso_sq = dso * dso
    
    for i in numba.prange(image_size):
        y = -(i - image_size / 2.0 + 0.5)
        x_start = 0 - image_size / 2.0 + 0.5
        
        for a in range(num_angles):
            cos_val = cos_a[a]
            sin_val = sin_a[a]
            
            x_prime = x_start * cos_val + y * sin_val
            y_prime = -x_start * sin_val + y * cos_val
            
            for j in range(image_size):
                dist_y = dso - y_prime
                
                if dist_y > 0:
                    inv_dist = 1.0 / dist_y
                    u = x_prime * dsd * inv_dist
                    u_idx = u / du + u_offset
                    
                    u_floor = int(u_idx)
                    if u_idx < 0:
                        u_floor -= 1
                        
                    if 0 <= u_floor < det_count - 1:
                        u_p = u_idx - u_floor
                        p1 = filtered_sino[a, u_floor]
                        p2 = filtered_sino[a, u_floor + 1]
                        
                        interp_val = p1 + u_p * (p2 - p1)
                        
                        weighting = dso_sq * inv_dist * inv_dist
                        reco[i, j] += interp_val * weighting
                
                x_prime += cos_val
                y_prime -= sin_val
                
    for i in numba.prange(image_size):
        for j in range(image_size):
            reco[i, j] *= dtheta
            
    return reco


def create_ramp_filter(det_count, du, filter_name='hann'):
    pad_len = 2 ** int(np.ceil(np.log2(2 * det_count)))
    
    n = np.arange(-pad_len // 2, pad_len // 2)
    h = np.zeros(pad_len, dtype=np.float32)
    
    h[n == 0] = 1.0 / (4 * du**2)
    odd = (n % 2 != 0)
    h[odd] = -1.0 / (np.pi**2 * n[odd]**2 * du**2)
    
    h_shifted = np.fft.ifftshift(h)
    H = np.fft.fft(h_shifted)
    
    if filter_name == 'none':
        # Simple backprojection without any filtering
        H = np.ones_like(H)
    else:
        # For all other filters, they act as windows multiplying the base Ram-Lak filter
        freqs = np.fft.fftfreq(pad_len, d=du)
        f_max = 1.0 / (2 * du)
        w = np.abs(freqs) / f_max  # Normalized frequency [0, 1]
        
        if filter_name == 'hann':
            window = 0.5 * (1 + np.cos(np.pi * w))
            H = H * window
        elif filter_name == 'hamming':
            window = 0.54 + 0.46 * np.cos(np.pi * w)
            H = H * window
        elif filter_name == 'cosine':
            window = np.cos(np.pi / 2.0 * w)
            H = H * window
        elif filter_name == 'shepp-logan':
            window = np.ones_like(w)
            non_zero = w > 0
            # Sinc function: sin(x)/x where x = (pi/2 * w)
            window[non_zero] = np.sin(np.pi / 2.0 * w[non_zero]) / (np.pi / 2.0 * w[non_zero])
            H = H * window
        
    return H, pad_len


def apply_filter_with_precomputed(sino, H, pad_len, du):
    angles, det_count = sino.shape
    
    sino_padded = np.zeros((angles, pad_len), dtype=np.float32)
    sino_padded[:, :det_count] = sino
    
    P = np.fft.fft(sino_padded, axis=1)
    Q = P * H
    
    q = np.real(np.fft.ifft(Q, axis=1)) * du
    return q[:, :det_count]


def run_custom_fbp(input_file, output_file, image_size=512, voxel_size=0.7, fbp_filter='hann'):
    print(f"Loading projections from {input_file} ...")
    projections, metadata = load_tiff_stack_with_metadata(Path(input_file))

    num_slices = projections.shape[2]
    print(f"Starting Optimized Custom Numba FBP reconstruction for {num_slices} slices...")
    
    reco = []
    total_start = time.time()
    
    # Extract native parameters (no hacks)
    vox_scaling = 1.0 / voxel_size
    dso_px = metadata['dso'] * vox_scaling
    ddo_px = metadata['ddo'] * vox_scaling
    dsd_px = dso_px + ddo_px
    du_px = metadata['du'] * vox_scaling
    det_count = projections.shape[1]
    # Add -pi/2 to rotate the reconstruction 90 degrees to match radiological convention
    angles = np.array(metadata['angles'])[:metadata['rotview']] - (np.pi / 2)
    
    # Precalculate the filter ONCE for all slices
    H, pad_len = create_ramp_filter(det_count, du_px, filter_name=fbp_filter)
    u_coords = (np.arange(det_count) - det_count / 2.0 + 0.5) * du_px
    weighting = dsd_px / np.sqrt(dsd_px**2 + u_coords**2)
    
    for i in tqdm.tqdm(range(num_slices), desc="Reconstructing Slices"):
        # We removed the np.flip() because standard FBP uses the native geometric orientation!
        prj = projections[:, :, i]
        sino = prj * vox_scaling
        
        # Step 1: Apply Cosine Weighting
        sino_weighted = sino * weighting
        
        # Step 2: Apply precomputed Ramp Filter
        filtered_sino = apply_filter_with_precomputed(sino_weighted, H, pad_len, du_px)
        
        # Step 3: Fast Numba Backprojection
        fbp = fan_beam_backprojection_ultra(filtered_sino, angles, image_size, dso_px, dsd_px, du_px, det_count)
        
        # Removed the arbitrary 0.5 multiplier to rely entirely on exact mathematical derivation
        reco.append(fbp)

    reco = np.array(reco)
    print(f"Total reconstruction time: {time.time() - total_start:.2f} seconds")

    print("Scaling reconstruction to Hounsfield Units (HU)...")
    hu_factor = metadata.get('hu_factor', 1.0) 
    fbp_hu = 1000 * ((reco - hu_factor) / hu_factor)

    print(f"Saving reconstruction to {output_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_to_tiff_stack(fbp_hu, Path(output_file))

    print("Reconstruction complete!")
    return projections, fbp_hu


if __name__ == '__main__':
    input_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_flat_fan_projections.tif'
    output_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_custom_reconstruction.tif'
    plot_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_custom_reconstruction_plot.png'
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found at {input_file}")
    else:
        proj, reco = run_custom_fbp(input_file, output_file)
        
        print("Generating a visual plot...")
        mid_slice = int(reco.shape[0] * 0.5)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]})
        
        axes[0].imshow(np.transpose(proj[:, :, int(proj.shape[2] * 0.5)]), cmap='gray')
        axes[0].set_title('Rebinned Projections (Fan-beam)', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(reco[mid_slice], cmap='gray', vmin=-300, vmax=300)
        axes[1].set_title(f'Ultra-Fast FBP (Slice {mid_slice})', fontsize=14)
        axes[1].axis('off')
        
        fig.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visual plot saved to {plot_file}")
