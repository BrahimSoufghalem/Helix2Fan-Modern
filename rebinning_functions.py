# ==========================================================================
# Modified from original helix2fan project under Apache License Version 2.0
# Modifications include:
# - Chronological DICOM sorting
# - High-performance vectorized rebinning 
# - Smart ASTRA/Numba FBP reconstruction pipeline (removed torch_radon)
# ==========================================================================

import numpy as np
import scipy.ndimage
import tqdm


def rebin_curved_to_flat_detector(args, proj_curved_helic):
    """ Rebin cylindrically curved detector projections to flat detector projections. Simultaneously, the central
    detector position (det_central_element) is shifted to the real geometric center of the curved detector.
    
    Vectorized version for high performance.

    :param args: required geometry parameters
    :param proj_curved_helic: curved detector projections
    :return: rebinned detector projections
    """
    proj_flat_helic = np.zeros_like(proj_curved_helic, dtype=np.float32)

    nu = args.nu
    nv = args.nv
    du = args.du
    dv = args.dv
    dsd = args.dsd

    i_x_det = np.arange(nu)
    i_z_det = np.arange(nv)

    X_det, Z_det = np.meshgrid(i_x_det, i_z_det)
    
    x_det = (X_det - nu / 2) * du + 0.5 * du
    z_det = (Z_det - nv / 2) * dv + 0.5 * dv

    norm = np.sqrt(x_det**2 + dsd**2 + z_det**2)

    p_on_curved_det_x = (x_det / norm) * dsd
    p_on_curved_det_z = (z_det / norm) * dsd

    phi_on_curved_det = np.arcsin(p_on_curved_det_x / dsd)
    dphi_curved = 2 * np.arctan(du / (2 * dsd))

    y_interp = phi_on_curved_det / dphi_curved + (nu - args.det_central_element[0])
    x_interp = p_on_curved_det_z / dv + (nv - args.det_central_element[1])

    coords = np.stack([x_interp, y_interp])

    for i_angle in tqdm.tqdm(range(proj_curved_helic.shape[0]), 'Rebin curved to flat detector'):
        proj_flat_helic[i_angle] = scipy.ndimage.map_coordinates(
            proj_curved_helic[i_angle],
            coords,
            order=1,
            mode='constant',
            cval=0.0
        )

    return proj_flat_helic


def _rebin_curved_to_flat_detector_core(args, proj_curved_helic, i_angle):
    """ Core loops to rebin cylindrically curved detector projections to flat detector projections. Simultaneously,
    the central detector position (det_central_element) is shifted to the real geometric center of the curved detector.
    
    Vectorized version.

    :param args: required geometry parameters
    :param proj_curved_helic: curved detector projections
    :param i_angle: index of rebinned projection for multiprocessing
    :return: rebinned detector projections
    """
    nu = args.nu
    nv = args.nv
    du = args.du
    dv = args.dv
    dsd = args.dsd

    i_x_det = np.arange(nu)
    i_z_det = np.arange(nv)

    X_det, Z_det = np.meshgrid(i_x_det, i_z_det)
    
    x_det = (X_det - nu / 2) * du + 0.5 * du
    z_det = (Z_det - nv / 2) * dv + 0.5 * dv

    norm = np.sqrt(x_det**2 + dsd**2 + z_det**2)

    p_on_curved_det_x = (x_det / norm) * dsd
    p_on_curved_det_z = (z_det / norm) * dsd

    phi_on_curved_det = np.arcsin(p_on_curved_det_x / dsd)
    dphi_curved = 2 * np.arctan(du / (2 * dsd))

    y_interp = phi_on_curved_det / dphi_curved + (nu - args.det_central_element[0])
    x_interp = p_on_curved_det_z / dv + (nv - args.det_central_element[1])

    coords = np.stack([x_interp, y_interp])

    proj_flat_helic_i_angle = scipy.ndimage.map_coordinates(
        proj_curved_helic[i_angle],
        coords,
        order=1,
        mode='constant',
        cval=0.0
    )

    return proj_flat_helic_i_angle


def rebin_curved_to_flat_detector_multiprocessing(data, cols):
    """ Function to rebin curved detector data to a flat panel. Needs to be called when multiprocessing with joblib.

    :param data: tuple of (args, proj_curved_helic)
    :param cols: angular index to iterate over projections
    :return: rebinned detector projections
    """
    args, proj_curved_helic = data

    return _rebin_curved_to_flat_detector_core(args, proj_curved_helic, cols)


def rebin_helical_to_fan_beam_trajectory(args, proj_helic):
    """ Rebin projections acquired on a helical trajectory to full-scan (2pi) fan beam projections.
    
    Vectorized version for high performance.

    :param args: required geometry parameters
    :param proj_helic: projections acquired on a helical trajectory
    :return: rebinned fan beam projections
    """
    distance = 0.5 * args.pitch  # Full scan. For short scan see Noo et al. "Single-slice rebinning ...".

    proj_rebinned = np.zeros((args.rotview, args.nu, args.nz_rebinned), dtype=np.float32)

    # Array of v locations of the detector elements relative to the central v detector element.
    # Consider 0.5 pixel shift.
    v_det_elements = (np.arange(0, args.nv, 1) - args.nv / 2 + 0.5) * args.dv
    v_det_min = v_det_elements[0]

    # Axial positions of resampled projections, starting at z_positions[0].
    z_poses_resampled = (np.arange(0, args.nz_rebinned, 1) * args.dv_rebinned) + args.z_positions[0]
    
    # Precompute u-related terms
    i_u_arr = np.arange(args.nu)
    u_vals = (i_u_arr - args.nu / 2 + 0.5) * args.du
    u_term1 = (u_vals**2 + args.dsd**2) / (args.dso * args.dsd)
    u_term2 = np.sqrt(u_vals**2 + args.dsd**2)
    u_vals_sq = u_vals**2

    # Loop over view angles.
    for s_angle in tqdm.tqdm(range(args.rotview), 'Rebin helical to fan-beam geometry'):
        # Find all valid projections at s_angle.
        z_poses_valid = args.z_positions[s_angle::args.rotview]

        for i_proj in range(len(z_poses_valid)):
            # Calculate indices of lower and upper z limit for each valid cone beam source position.
            lower_lim = z_poses_valid[i_proj] - distance
            upper_lim = z_poses_valid[i_proj] + distance

            i_lower_lim = np.clip(int((lower_lim - args.z_positions[0]) / args.dv_rebinned),
                                  a_min=0, a_max=len(z_poses_resampled))
            i_upper_lim = np.clip(int(np.ceil((upper_lim - args.z_positions[0]) / args.dv_rebinned)),
                                  a_min=0, a_max=len(z_poses_resampled))
            i_lower_lim, i_upper_lim = min(i_lower_lim, i_upper_lim), max(i_lower_lim, i_upper_lim)
            
            if i_lower_lim >= i_upper_lim:
                continue
                
            i_z_slice = slice(i_lower_lim, i_upper_lim)
            z_res_slice = z_poses_resampled[i_z_slice]

            # Axial distance between virtual and helical projection.
            deltaZ = z_poses_valid[i_proj] - z_res_slice
            
            # v_precise has shape (len(z_res_slice), args.nu)
            v_precise = np.outer(deltaZ, u_term1)
            
            # Convert v_precise to continuous index coordinates for map_coordinates
            v_idx = (v_precise - v_det_min) / args.dv
            
            # Generate u_idx of same shape
            U_idx, _ = np.meshgrid(i_u_arr, np.arange(len(z_res_slice)))
            
            coords = np.stack([v_idx, U_idx])
            
            # Extract projection slice: shape (args.nv, args.nu)
            v_det_values = proj_helic[s_angle + i_proj * args.rotview, :, :]
            
            # Interpolate (using mode='nearest' to match np.interp's edge duplication)
            v_interp = scipy.ndimage.map_coordinates(
                v_det_values,
                coords,
                order=1,
                mode='nearest'
            )
            
            # Eq.(1) from Noo et al.
            weight = u_term2[np.newaxis, :] / np.sqrt(u_vals_sq[np.newaxis, :] + v_precise**2 + args.dsd**2)
            
            proj_rebinned[s_angle, :, i_z_slice] = (weight * v_interp).T

    return proj_rebinned
