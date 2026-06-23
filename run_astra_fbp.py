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
import astra
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

def run_astra_fbp(input_file, output_file, image_size=512, voxel_size=0.7, filter_type='Hann'):
    print(f"Loading projections from {input_file} ...")
    projections, metadata = load_tiff_stack_with_metadata(Path(input_file))

    num_slices = projections.shape[2]
    print(f"Starting ASTRA Toolbox GPU FBP reconstruction for {num_slices} slices...")
    
    reco_volume = []
    total_start = time.time()
    
    # Scale real-world mm to pixels
    vox_scaling = 1.0 / voxel_size
    dso_px = metadata['dso'] * vox_scaling
    ddo_px = metadata['ddo'] * vox_scaling
    du_px = metadata['du'] * vox_scaling
    det_count = projections.shape[1]
    
    # ASTRA expects counter-clockwise rotation, but our data is clockwise. We must negate the angles!
    # And we keep the -pi/2 offset to orient the spine correctly at the bottom.
    raw_angles = np.array(metadata['angles'])[:metadata['rotview']]
    angles = -raw_angles - (np.pi / 2)
    
    # 1. Create Volume Geometry
    # This defines a 2D slice of size (image_size x image_size)
    vol_geom = astra.create_vol_geom(image_size, image_size)
    
    # 2. Create Projection Geometry (Fan-beam flat detector)
    # ASTRA params: 'fanflat', det_width, det_count, angles, source_origin, origin_det
    proj_geom = astra.create_proj_geom('fanflat', du_px, det_count, angles, dso_px, ddo_px)
    
    for i in tqdm.tqdm(range(num_slices), desc="Reconstructing Slices (ASTRA GPU)"):
        # ASTRA expects C-contiguous float32 arrays!
        sino = np.ascontiguousarray(projections[:, :, i] * vox_scaling, dtype=np.float32)
        
        # 3. Load data into ASTRA memory
        sino_id = astra.data2d.create('-sino', proj_geom, sino)
        reco_id = astra.data2d.create('-vol', vol_geom)
        
        # 4. Set up FBP_CUDA algorithm
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sino_id
        cfg['FilterType'] = filter_type  # ASTRA applies Ram-Lak or Hann natively on GPU
        
        alg_id = astra.algorithm.create(cfg)
        
        # 5. Run it on the GPU
        astra.algorithm.run(alg_id)
        
        # 6. Retrieve the reconstructed slice
        fbp_slice = astra.data2d.get(reco_id)
        
        # ASTRA's coordinate handedness produces a mirrored image relative to the standard convention.
        # We simply flip the slice horizontally to place the heart on the correct side (right side of image).
        fbp_slice = np.flip(fbp_slice, axis=1)
        
        reco_volume.append(fbp_slice)
        
        # 7. Free GPU memory for this slice
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sino_id)

    reco_volume = np.array(reco_volume)
    print(f"Total GPU reconstruction time: {time.time() - total_start:.2f} seconds")

    print("Scaling reconstruction to Hounsfield Units (HU)...")
    # Depending on ASTRA's internal scaling, hu_factor might need a multiplier. 
    # But mathematically, FBP scales linearly with projection intensity.
    hu_factor = metadata.get('hu_factor', 1.0) 
    fbp_hu = 1000 * ((reco_volume - hu_factor) / hu_factor)

    print(f"Saving reconstruction to {output_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_to_tiff_stack(fbp_hu, Path(output_file))

    print("Reconstruction complete!")
    return projections, fbp_hu

if __name__ == '__main__':
    input_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_flat_fan_projections.tif'
    output_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_astra_reconstruction.tif'
    plot_file = '/home/BArch/.gemini/antigravity/scratch/helix2fan/out/scan_001_astra_reconstruction_plot.png'
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found at {input_file}")
        print("Please run main.py first to generate the projection data.")
    else:
        try:
            proj, reco = run_astra_fbp(input_file, output_file)
            
            print("Generating a visual plot...")
            mid_slice = int(reco.shape[0] * 0.5)
            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]})
            
            axes[0].imshow(np.transpose(proj[:, :, int(proj.shape[2] * 0.5)]), cmap='gray')
            axes[0].set_title('Rebinned Projections (Fan-beam)', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(reco[mid_slice], cmap='gray', vmin=-300, vmax=300)
            axes[1].set_title(f'ASTRA GPU FBP (Slice {mid_slice})', fontsize=14)
            axes[1].axis('off')
            
            fig.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Visual plot saved to {plot_file}")
            
        except ImportError as e:
            print("ASTRA Toolbox is not installed.")
            print("Please install it using: conda install -c astra-toolbox astra-toolbox")
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            print("Ensure you have a CUDA-capable GPU configured correctly.")
