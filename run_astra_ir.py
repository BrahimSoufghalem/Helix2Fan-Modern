# ==========================================================================
# Part of helix2fan_improved — Apache License Version 2.0
# GPU-accelerated Iterative Reconstruction via ASTRA Toolbox
# Algorithms: SIRT, SART, CGLS, TV-SIRT
# ==========================================================================

import numpy as np
from pathlib import Path
import astra
import tqdm
import time

from helper import load_tiff_stack_with_metadata, save_to_tiff_stack


# ─────────────────────────────────────────────────────────────────────────────
# Total Variation (TV) Denoising — Gradient Descent on TV functional
# Applied between SIRT iterations in the TV-SIRT algorithm.
# ─────────────────────────────────────────────────────────────────────────────
def _tv_denoising_step(img: np.ndarray, lam: float, n_iter: int = 5) -> np.ndarray:
    """
    Applies n_iter steps of gradient descent on the isotropic Total Variation
    functional to promote piece-wise constant regions (edge-preserving smoothing).

    Args:
        img    : 2D image array to denoise.
        lam    : TV regularization strength. Higher values = smoother image.
        n_iter : Number of inner gradient-descent steps per call.
    Returns:
        Denoised 2D image array.
    """
    u = img.copy().astype(np.float32)
    for _ in range(n_iter):
        # Forward differences (gradient)
        gx = np.roll(u, -1, axis=1) - u
        gy = np.roll(u, -1, axis=0) - u

        # Normalise to get unit subgradient (eps avoids division by zero)
        denom = np.sqrt(gx**2 + gy**2 + 1e-8)
        px = gx / denom
        py = gy / denom

        # Divergence (backward differences)
        div = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))

        # Gradient descent step
        u = u + lam * div
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Single-slice Iterative Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def _reconstruct_slice_ir(
    sino_slice: np.ndarray,
    proj_geom: dict,
    vol_geom: dict,
    method: str,
    iterations: int,
    tv_lambda: float,
    tv_inner_iters: int,
) -> np.ndarray:
    """
    Reconstruct one 2D slice using the requested iterative algorithm.

    Args:
        sino_slice    : 2D sinogram array (views × detectors), float32, C-contiguous.
        proj_geom     : ASTRA projection geometry dict.
        vol_geom      : ASTRA volume geometry dict.
        method        : One of 'sirt', 'sart', 'cgls', 'tv-sirt'.
        iterations    : Total number of main iterations.
        tv_lambda     : TV regularisation weight (only for tv-sirt).
        tv_inner_iters: TV gradient-descent steps per outer iteration (only for tv-sirt).
    Returns:
        2D reconstructed slice as float32 ndarray.
    """
    sino_id = astra.data2d.create('-sino', proj_geom, sino_slice)
    reco_id = astra.data2d.create('-vol', vol_geom)

    if method == 'sirt':
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        astra.algorithm.delete(alg_id)

    elif method == 'sart':
        cfg = astra.astra_dict('SART_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        astra.algorithm.delete(alg_id)

    elif method == 'cgls':
        cfg = astra.astra_dict('CGLS_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        astra.algorithm.delete(alg_id)

    elif method == 'tv-sirt':
        # TV-SIRT: interleave SIRT steps with TV denoising passes.
        # We split `iterations` across `OUTER_ITERS` outer rounds,
        # each running (iterations // OUTER_ITERS) SIRT steps followed
        # by one TV denoising pass.
        OUTER_ITERS = 20
        sirt_per_round = max(1, iterations // OUTER_ITERS)

        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)

        for _ in range(OUTER_ITERS):
            # SIRT update
            astra.algorithm.run(alg_id, sirt_per_round)
            # TV denoising on the current estimate
            current = astra.data2d.get(reco_id)
            smoothed = _tv_denoising_step(current, tv_lambda, tv_inner_iters)
            astra.data2d.store(reco_id, smoothed)

        astra.algorithm.delete(alg_id)

    else:
        raise ValueError(
            f"Unknown IR method '{method}'. "
            "Choose from: 'sirt', 'sart', 'cgls', 'tv-sirt'."
        )

    result = np.array(astra.data2d.get(reco_id), dtype=np.float32)

    # Free GPU memory immediately — critical for large volumes
    astra.data2d.delete(sino_id)
    astra.data2d.delete(reco_id)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def run_astra_ir(
    input_file: str,
    output_file: str,
    image_size: int = 512,
    voxel_size: float = 0.7,
    method: str = 'sirt',
    iterations: int = 100,
    tv_lambda: float = 0.01,
    tv_inner_iters: int = 5,
):
    """
    Iterative CT reconstruction of a full volume using ASTRA Toolbox (GPU).

    Args:
        input_file    : Path to the rebinned fan-beam sinogram (.tif stack).
        output_file   : Destination path for the reconstructed volume (.tif stack).
        image_size    : Reconstructed image side length in pixels (default 512).
        voxel_size    : Physical voxel size in mm (default 0.7).
        method        : IR algorithm — 'sirt', 'sart', 'cgls', or 'tv-sirt'.
        iterations    : Number of main algorithm iterations.
        tv_lambda     : TV regularisation weight (only used by tv-sirt).
        tv_inner_iters: TV gradient-descent steps per outer iteration (tv-sirt only).
    """
    print(f"\n{'='*60}")
    print(f"  Iterative Reconstruction  |  Method: {method.upper()}")
    print(f"  Iterations: {iterations}"
          + (f"  |  TV λ: {tv_lambda}" if method == 'tv-sirt' else ""))
    print(f"{'='*60}\n")

    print(f"Loading projections from {input_file} ...")
    projections, metadata = load_tiff_stack_with_metadata(Path(input_file))

    num_slices = projections.shape[2]
    print(f"Volume: {projections.shape[0]} views × "
          f"{projections.shape[1]} det × {num_slices} slices\n")

    # ── Geometry (mm → pixels) ────────────────────────────────────────────
    vox_scaling = 1.0 / voxel_size
    dso_px = metadata['dso'] * vox_scaling
    ddo_px = metadata['ddo'] * vox_scaling
    du_px  = metadata['du']  * vox_scaling
    det_count = projections.shape[1]

    # Negate angles: ASTRA uses CCW, our data is CW.
    # Keep -π/2 offset for correct anatomical orientation (spine at bottom).
    raw_angles = np.array(metadata['angles'])[:metadata['rotview']]
    angles = -raw_angles - (np.pi / 2)

    # ── ASTRA Geometry Objects (created once, reused for all slices) ──────
    vol_geom  = astra.create_vol_geom(image_size, image_size)
    proj_geom = astra.create_proj_geom(
        'fanflat', du_px, det_count, angles, dso_px, ddo_px
    )

    # ── Reconstruct slice by slice ────────────────────────────────────────
    reco_volume = []
    total_start = time.time()

    for i in tqdm.tqdm(range(num_slices), desc=f"IR [{method.upper()}]"):
        sino_slice = np.ascontiguousarray(
            projections[:, :, i] * vox_scaling, dtype=np.float32
        )

        recon = _reconstruct_slice_ir(
            sino_slice, proj_geom, vol_geom,
            method=method,
            iterations=iterations,
            tv_lambda=tv_lambda,
            tv_inner_iters=tv_inner_iters,
        )

        # Mirror correction (same as FBP) to match radiological convention
        reco_volume.append(np.flip(recon, axis=1))

    elapsed = time.time() - total_start
    print(f"\nTotal reconstruction time: {elapsed:.1f}s "
          f"({elapsed/num_slices:.2f}s per slice)")

    # ── Convert to Hounsfield Units ───────────────────────────────────────
    reco_volume = np.array(reco_volume, dtype=np.float32)
    hu_factor   = metadata.get('hu_factor', 1.0)
    reco_hu     = 1000.0 * ((reco_volume - hu_factor) / hu_factor)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Saving reconstruction to {output_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_to_tiff_stack(reco_hu, Path(output_file))

    print(f"\n✅ Iterative Reconstruction [{method.upper()}] complete!")
    return reco_volume, reco_hu
