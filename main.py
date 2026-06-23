# ==========================================================================
# Modified from original helix2fan project under Apache License Version 2.0
# Modifications include:
# - Chronological DICOM sorting
# - High-performance vectorized rebinning 
# - Smart ASTRA/Numba FBP reconstruction pipeline (removed torch_radon)
# ==========================================================================

import numpy as np
import argparse
import tqdm
from pathlib import Path
from helper import save_to_tiff_stack_with_metadata, load_tiff_stack_with_metadata
from rebinning_functions import rebin_curved_to_flat_detector, rebin_helical_to_fan_beam_trajectory
from read_data import read_dicom


def run(parser):
    args = parser.parse_args()
    print('Processing scan {}.'.format(args.scan_id))

    # Load projections and read out geometry data from the DICOM header.
    raw_projections, parser = read_dicom(parser)
    args = parser.parse_args()

    if args.save_all:
        save_path = Path(args.path_out) / Path('{}_curved_helix_projections.tif'.format(args.scan_id))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_tiff_stack_with_metadata(raw_projections,
                                         save_path,
                                         metadata=vars(args))

    # Rebin helical projections from curved detector to flat detector.
    # Uses the highly optimized vectorized version.
    proj_flat_detector = rebin_curved_to_flat_detector(args, raw_projections)

    if args.save_all:
        save_path = Path(args.path_out) / Path('{}_flat_helix_projections.tif'.format(args.scan_id))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_tiff_stack_with_metadata(proj_flat_detector,
                                         save_path,
                                         metadata=vars(args))

    # Rebinning of projections acquired on a helical trajectory to full-scan (2pi) fan beam projections.
    proj_fan_geometry = rebin_helical_to_fan_beam_trajectory(args, proj_flat_detector)

    flat_fan_path = Path(args.path_out) / Path('{}_flat_fan_projections.tif'.format(args.scan_id))
    flat_fan_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_tiff_stack_with_metadata(proj_fan_geometry,
                                     flat_fan_path,
                                     metadata=vars(args))

    print('Finished Rebinning. Results saved at {}.'.format(flat_fan_path.resolve()))
    
    # --- Intelligent Reconstruction Pipeline ---
    print("\n" + "="*60)
    print("   Initializing Smart FBP Reconstruction Pipeline")
    print("="*60)
    
    output_file = Path(args.path_out) / Path('{}_reconstruction.tif'.format(args.scan_id))
    
    # ── Choose reconstruction engine ──────────────────────────────────────
    astra_available = False
    try:
        import astra
        astra_available = True
    except ImportError:
        pass

    is_ir = args.reco_method in ('sirt', 'sart', 'cgls', 'tv-sirt')

    if is_ir:
        # Iterative Reconstruction always requires ASTRA (GPU)
        if not astra_available:
            print("❌ Iterative Reconstruction requires ASTRA Toolbox (GPU).")
            print("   Install with: conda install -c astra-toolbox astra-toolbox")
            print("   Falling back to CPU FBP...")
            from run_custom_fbp import run_custom_fbp
            run_custom_fbp(str(flat_fan_path), str(output_file), fbp_filter=args.fbp_filter)
        else:
            print(f"🔬 ASTRA Toolbox detected! Launching {args.reco_method.upper()} iterative reconstruction...")
            from run_astra_ir import run_astra_ir
            try:
                run_astra_ir(
                    str(flat_fan_path), str(output_file),
                    method=args.reco_method,
                    iterations=args.iterations,
                    tv_lambda=args.tv_lambda,
                )
            except Exception as e:
                print(f"⚠️ IR Reconstruction failed ({e}). Falling back to CPU FBP...")
                from run_custom_fbp import run_custom_fbp
                run_custom_fbp(str(flat_fan_path), str(output_file), fbp_filter=args.fbp_filter)
    else:
        # FBP path (default)
        if astra_available:
            print("🚀 ASTRA Toolbox detected! Launching GPU-accelerated FBP reconstruction...")
            from run_astra_fbp import run_astra_fbp
            try:
                run_astra_fbp(str(flat_fan_path), str(output_file), filter_type=args.fbp_filter)
            except Exception as e:
                print(f"⚠️ GPU FBP failed ({e}). Falling back to CPU...")
                from run_custom_fbp import run_custom_fbp
                run_custom_fbp(str(flat_fan_path), str(output_file), fbp_filter=args.fbp_filter)
        else:
            print("💻 ASTRA Toolbox not found. Launching Ultra-Fast Numba CPU FBP...")
            from run_custom_fbp import run_custom_fbp
            run_custom_fbp(str(flat_fan_path), str(output_file), fbp_filter=args.fbp_filter)
        
    print("\n✅ End-to-end pipeline completed successfully!")

    if getattr(args, 'plot_result', 'none') != 'none':
        try:
            import matplotlib.pyplot as plt
            print("\n📊 Generating visualization...")
            sino_stack, _ = load_tiff_stack_with_metadata(flat_fan_path)
            reco_stack, _ = load_tiff_stack_with_metadata(output_file)
            
            slice_idx = getattr(args, 'plot_slice', -1)
            if slice_idx == -1:
                idx_sino = sino_stack.shape[0] // 2
                idx_reco = reco_stack.shape[0] // 2
            else:
                idx_sino = min(slice_idx, sino_stack.shape[0] - 1)
                idx_reco = min(slice_idx, reco_stack.shape[0] - 1)
            
            sino_img = sino_stack[idx_sino]
            reco_img = reco_stack[idx_reco]

            plot_save_path = output_file.parent / f"{args.scan_id}_visualization.png"

            if args.plot_result == 'both':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(sino_img, cmap='gray', aspect='auto')
                ax1.set_title('Sinogram (Central Slice)')
                ax1.axis('off')
                
                ax2.imshow(reco_img, cmap='gray', vmin=0, vmax=np.percentile(reco_img, 99))
                ax2.set_title('Reconstructed Slice')
                ax2.axis('off')
                plt.tight_layout()
                plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
                print(f"🖼️ Visualization saved to: {plot_save_path}")
                plt.show()
                
            elif args.plot_result == 'sinogram':
                plt.figure(figsize=(8, 6))
                plt.imshow(sino_img, cmap='gray', aspect='auto')
                plt.title('Sinogram (Central Slice)')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
                print(f"🖼️ Visualization saved to: {plot_save_path}")
                plt.show()
                
            elif args.plot_result == 'reconstruction':
                plt.figure(figsize=(8, 8))
                plt.imshow(reco_img, cmap='gray', vmin=0, vmax=np.percentile(reco_img, 99))
                plt.title('Reconstructed Slice')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
                print(f"🖼️ Visualization saved to: {plot_save_path}")
                plt.show()
        except Exception as e:
            print(f"⚠️ Could not plot results: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dicom', type=str, required=True, help='Local path of helical projection data.')
    parser.add_argument('--path_out', type=str, default='out', help='Output path of rebinned data.')
    parser.add_argument('--scan_id', type=str, default='scan_001', help='Custom scan ID.')
    parser.add_argument('--idx_proj_start', type=int, default=12000, help='First index of helical projections that are processed.')
    parser.add_argument('--idx_proj_stop', type=int, default=16000, help='Last index of helical projections that are processed.')
    # ── Reconstruction ────────────────────────────────────────────────────
    parser.add_argument('--reco_method', type=str, default='fbp',
                        choices=['fbp', 'sirt', 'sart', 'cgls', 'tv-sirt'],
                        help='Reconstruction method: fbp (fast), sirt/sart/cgls/tv-sirt (iterative, GPU required).')
    parser.add_argument('--fbp_filter', type=str, default='hann',
                        help='FBP filter type (fbp mode only). Options: hann, hamming, shepp-logan, ram-lak, none.')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations (IR methods only, default: 100).')
    parser.add_argument('--tv_lambda', type=float, default=0.01,
                        help='TV regularisation strength (tv-sirt only, default: 0.01).')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='Save all intermediate results.')
    parser.add_argument('--plot_result', type=str, default='both',
                        choices=['none', 'sinogram', 'reconstruction', 'both'],
                        help='Automatically display a plot after reconstruction: "both" (default), "sinogram", "reconstruction", or "none".')
    parser.add_argument('--plot_slice', type=int, default=-1,
                        help='Specific slice index to plot. Default is -1 (which automatically selects the middle slice).')
    run(parser)
