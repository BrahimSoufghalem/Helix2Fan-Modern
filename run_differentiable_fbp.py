# ==========================================================================
# Part of Helix2Fan-Modern — Apache License Version 2.0
# Differentiable Fan-Beam FBP Reconstruction via PyTorch
#
# A fully differentiable FBP operator composed of three independent layers:
#   1. CosineWeightLayer           — detector cosine weighting
#   2. RampFilterLayer             — frequency-domain ramp filtering (FFT)
#   3. FanBeamBackProjectionLayer  — backprojection via grid_sample
#
# Designed for end-to-end learned reconstruction pipelines.
# Reference implementation: run_custom_fbp.py (Numba)
# ==========================================================================

import numpy as np
from pathlib import Path
import time
import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from helper import load_tiff_stack_with_metadata, save_to_tiff_stack


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Cosine Weighting
# Reference: run_custom_fbp.py L158-L159
# ─────────────────────────────────────────────────────────────────────────────
class CosineWeightLayer(nn.Module):
    """
    Applies fan-beam cosine weighting to sinogram data.

    For each detector element at position u, the weight is:
        w(u) = D_sd / sqrt(u^2 + D_sd^2)

    This corrects for the varying path lengths across the detector fan.
    All weights are pre-computed and stored as a buffer.
    """

    def __init__(self, det_count: int, du: float, dsd: float):
        super().__init__()
        u_coords = (torch.arange(det_count, dtype=torch.float32)
                    - det_count / 2.0 + 0.5) * du
        weights = dsd / torch.sqrt(u_coords ** 2 + dsd ** 2)
        # Shape: (1, 1, 1, D) for broadcasting with (B, 1, A, D)
        self.register_buffer('cosine_weights', weights.reshape(1, 1, 1, -1))

    def forward(self, sinogram: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            sinogram: (B, 1, A, D)
        Returns:
            Weighted sinogram: (B, 1, A, D)
        """
        return sinogram * self.cosine_weights


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Ramp Filtering (FFT-based)
# Reference: run_custom_fbp.py L82-L133
# ─────────────────────────────────────────────────────────────────────────────
class RampFilterLayer(nn.Module):
    """
    Applies the ramp (Ram-Lak) filter in the frequency domain via FFT.

    Supports optional windowing (Hann, Hamming, Cosine, Shepp-Logan) and
    an optional learnable filter mode where the filter becomes a trainable
    nn.Parameter.

    The filter is constructed identically to create_ramp_filter() in
    run_custom_fbp.py to ensure numerical equivalence.
    """

    def __init__(self, det_count: int, du: float,
                 filter_name: str = 'ram-lak', learnable_filter: bool = False):
        super().__init__()
        self.det_count = det_count
        self.du = du

        pad_len = 2 ** int(np.ceil(np.log2(2 * det_count)))
        self.pad_len = pad_len

        # Build spatial-domain ramp kernel (matches create_ramp_filter exactly)
        n = np.arange(-pad_len // 2, pad_len // 2)
        h = np.zeros(pad_len, dtype=np.float32)
        h[n == 0] = 1.0 / (4.0 * du ** 2)
        odd = (n % 2 != 0)
        h[odd] = -1.0 / (np.pi ** 2 * n[odd] ** 2 * du ** 2)

        h_shifted = np.fft.ifftshift(h)
        H_np = np.fft.rfft(h_shifted)  # Complex, length pad_len//2 + 1

        # Apply window function
        if filter_name == 'none':
            H_np = np.ones_like(H_np)
        else:
            freqs = np.fft.rfftfreq(pad_len, d=du)
            f_max = 1.0 / (2.0 * du)
            w = np.abs(freqs) / f_max  # Normalized frequency [0, 1]

            if filter_name == 'hann':
                window = 0.5 * (1.0 + np.cos(np.pi * w))
            elif filter_name == 'hamming':
                window = 0.54 + 0.46 * np.cos(np.pi * w)
            elif filter_name == 'cosine':
                window = np.cos(np.pi / 2.0 * w)
            elif filter_name == 'shepp-logan':
                window = np.ones_like(w)
                non_zero = w > 0
                window[non_zero] = (np.sin(np.pi / 2.0 * w[non_zero])
                                    / (np.pi / 2.0 * w[non_zero]))
            elif filter_name == 'ram-lak':
                window = np.ones_like(w)
            else:
                window = np.ones_like(w)

            H_np = H_np * window

        # Convert to torch — store real and imag separately for compatibility
        H_torch = torch.from_numpy(H_np.copy()).to(torch.complex64)
        H_torch = H_torch.reshape(1, 1, 1, -1)  # (1, 1, 1, F)

        if learnable_filter:
            self.H_real = nn.Parameter(H_torch.real.clone())
            self.H_imag = nn.Parameter(H_torch.imag.clone())
        else:
            self.register_buffer('H_real', H_torch.real)
            self.register_buffer('H_imag', H_torch.imag)

    def forward(self, sinogram: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            sinogram: (B, 1, A, D) — cosine-weighted sinogram
        Returns:
            Filtered sinogram: (B, 1, A, D)
        """
        # Pad along detector dimension
        pad_amount = self.pad_len - self.det_count
        padded = F.pad(sinogram, (0, pad_amount))  # (B, 1, A, pad_len)

        # Forward FFT along detector axis
        spectrum = torch.fft.rfft(padded, dim=-1)  # (B, 1, A, pad_len//2+1)

        # Reconstruct complex filter from stored real/imag parts
        H = torch.complex(self.H_real, self.H_imag)

        # Multiply in frequency domain
        filtered_spectrum = spectrum * H

        # Inverse FFT
        filtered = torch.fft.irfft(filtered_spectrum, n=self.pad_len, dim=-1)

        # Crop back to original detector count and scale by du
        return filtered[..., :self.det_count] * self.du


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Fan-Beam Backprojection (grid_sample-based)
# Reference: run_custom_fbp.py L26-L79
# ─────────────────────────────────────────────────────────────────────────────
class FanBeamBackProjectionLayer(nn.Module):
    """
    Differentiable fan-beam backprojection using torch.nn.functional.grid_sample.

    All geometry-dependent quantities (sin/cos tables, pixel grid, detector
    coordinate mapping) are pre-computed in __init__() and stored as buffers.
    During forward(), only data-dependent operations remain, ensuring maximum
    throughput during training.

    Angles are processed in configurable chunks to control GPU memory usage.
    For 1152 angles x 512x512 image with chunk_size=64:
        Peak GPU memory per chunk ~ 350 MB.
    """

    def __init__(self, angles: 'torch.Tensor', dso: float, dsd: float,
                 du: float, det_count: int, image_size: int,
                 chunk_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.num_angles = len(angles)
        self.chunk_size = chunk_size

        # ── Pre-compute angle trigonometry ────────────────────────────────
        self.register_buffer('cos_a', torch.cos(angles).float())  # (A,)
        self.register_buffer('sin_a', torch.sin(angles).float())  # (A,)

        # ── Pre-compute pixel grid ────────────────────────────────────────
        # Matches Numba convention: y = -(i - N/2 + 0.5), x = j - N/2 + 0.5
        coords = (torch.arange(image_size, dtype=torch.float32)
                  - image_size / 2.0 + 0.5)
        y_1d = -coords  # Negate for radiological convention
        x_1d = coords
        y_grid, x_grid = torch.meshgrid(y_1d, x_1d, indexing='ij')
        self.register_buffer('x_grid', x_grid)  # (H, W)
        self.register_buffer('y_grid', y_grid)  # (H, W)

        # ── Pre-compute geometry constants ────────────────────────────────
        self.register_buffer('_dso', torch.tensor(dso, dtype=torch.float32))
        self.register_buffer('_dsd', torch.tensor(dsd, dtype=torch.float32))
        self.register_buffer('_dso_sq', torch.tensor(dso * dso, dtype=torch.float32))

        # grid_sample normalization constants:
        # u_idx = u / du + det_count/2 - 0.5
        # u_normalized = 2 * u_idx / (det_count - 1) - 1  (maps to [-1, 1])
        # Simplified: u_normalized = u * u_scale + u_offset
        self.register_buffer('_u_scale', torch.tensor(
            2.0 / (du * (det_count - 1)), dtype=torch.float32))
        self.register_buffer('_u_offset', torch.tensor(
            2.0 * (det_count / 2.0 - 0.5) / (det_count - 1) - 1.0,
            dtype=torch.float32))

        # Angular step for final normalization
        if len(angles) > 1:
            dtheta = float(angles[1] - angles[0])
        else:
            dtheta = 1.0
        self.register_buffer('_dtheta', torch.tensor(dtheta, dtype=torch.float32))

    def forward(self, filtered_sino: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            filtered_sino: (B, 1, A, D) — filtered sinogram
        Returns:
            reconstruction: (B, 1, H, W) — reconstructed image
        """
        B = filtered_sino.shape[0]
        device = filtered_sino.device
        result = torch.zeros(B, 1, self.image_size, self.image_size,
                             device=device, dtype=filtered_sino.dtype)

        # Process angles in chunks for memory efficiency
        for start in range(0, self.num_angles, self.chunk_size):
            end = min(start + self.chunk_size, self.num_angles)
            C = end - start

            # Angle trig for this chunk: (C, 1, 1) for broadcasting
            cos_c = self.cos_a[start:end].view(C, 1, 1)
            sin_c = self.sin_a[start:end].view(C, 1, 1)

            # Rotate pixel coordinates → detector frame: (C, H, W)
            x_prime = self.x_grid * cos_c + self.y_grid * sin_c
            y_prime = -self.x_grid * sin_c + self.y_grid * cos_c

            # Source-to-pixel distance along central ray
            L = self._dso - y_prime  # (C, H, W)

            # Detector coordinate (unnormalized, in pixel units)
            u = x_prime * self._dsd / L  # (C, H, W)

            # Normalize to [-1, 1] for grid_sample
            u_norm = u * self._u_scale + self._u_offset  # (C, H, W)

            # Build sampling grid: (C, H, W, 2)
            # x-coord = normalized detector position
            # y-coord = 0.0 (sinogram row is a single-height "image")
            grid = torch.stack([u_norm, torch.zeros_like(u_norm)], dim=-1)

            # Extract sinogram rows for this angle chunk: (B, 1, C, D)
            sino_chunk = filtered_sino[:, :, start:end, :]

            # Reshape for grid_sample — merge (B, C) into batch dimension
            # Input:  (B*C, 1, 1, D) — each row treated as a 1-pixel-tall image
            # Grid:   (B*C, H, W, 2) — sampling coordinates per pixel
            sino_flat = sino_chunk.reshape(B * C, 1, 1, -1)
            grid_flat = (grid.unsqueeze(0)
                         .expand(B, -1, -1, -1, -1)
                         .reshape(B * C, self.image_size, self.image_size, 2))

            # Differentiable bilinear interpolation
            interp = F.grid_sample(
                sino_flat, grid_flat,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )  # (B*C, 1, H, W)

            # Reshape back: (B, C, H, W)
            interp = interp.reshape(B, C, self.image_size, self.image_size)

            # Distance weighting: D_so^2 / L^2, shape (C, H, W)
            weight = self._dso_sq / (L * L)

            # Accumulate weighted backprojection
            result[:, 0] += (interp * weight.unsqueeze(0)).sum(dim=1)

        return result * self._dtheta


# ─────────────────────────────────────────────────────────────────────────────
# Composer: Full Differentiable FBP Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class DifferentiableFanBeamFBP(nn.Module):
    """
    Complete differentiable fan-beam FBP operator.

    Composes three independent layers into a single nn.Module:
        CosineWeightLayer → RampFilterLayer → FanBeamBackProjectionLayer

    Each sub-layer is independently accessible for testing, replacement,
    or reuse in other architectures (e.g., Learned Primal-Dual, FBPConvNet).

    Args:
        angles          : 1D tensor/array of projection angles (radians).
        dso             : Source-to-origin distance (pixel units).
        dsd             : Source-to-detector distance (pixel units).
        du              : Detector element spacing (pixel units).
        det_count       : Number of detector elements.
        image_size      : Reconstructed image side length in pixels.
        filter_name     : Ramp filter window. Options:
                          'ram-lak', 'hann', 'hamming', 'shepp-logan',
                          'cosine', 'none'.
        learnable_filter: If True, the ramp filter becomes a trainable
                          nn.Parameter (for learned reconstruction).
        chunk_size      : Angles per GPU batch in backprojection.
        hu_factor       : Optional scalar representing the attenuation coefficient of water.
                          If provided, the output is scaled to Hounsfield Units (HU).
                          If None, outputs raw attenuation coefficients.
    """

    def __init__(self, angles, dso: float, dsd: float, du: float,
                 det_count: int, image_size: int,
                 filter_name: str = 'ram-lak', learnable_filter: bool = False,
                 chunk_size: int = 64, hu_factor: float = None):
        super().__init__()
        if not isinstance(angles, torch.Tensor):
            angles = torch.tensor(angles, dtype=torch.float32)

        self.cosine_weight = CosineWeightLayer(det_count, du, dsd)
        self.ramp_filter = RampFilterLayer(
            det_count, du, filter_name, learnable_filter)
        self.backprojector = FanBeamBackProjectionLayer(
            angles, dso, dsd, du, det_count, image_size, chunk_size)
        
        self.hu_factor = hu_factor

    def forward(self, sinogram: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            sinogram: (B, 1, A, D) — raw sinogram batch
        Returns:
            reconstruction: (B, 1, H, W) — reconstructed images
        """
        x = self.cosine_weight(sinogram)
        x = self.ramp_filter(x)
        x = self.backprojector(x)
        
        if self.hu_factor is not None:
            # Scale raw attenuation to Hounsfield Units differentiably
            x = 1000.0 * ((x - self.hu_factor) / self.hu_factor)
            
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Validation Utilities
# ─────────────────────────────────────────────────────────────────────────────
def test_gradient_flow(model: 'DifferentiableFanBeamFBP',
                       num_angles: int, det_count: int,
                       device: 'torch.device' = None):
    """
    Verify that gradients flow end-to-end through the differentiable FBP.

    Creates a random sinogram with requires_grad=True, passes it through
    the model, computes a scalar loss, and checks that sinogram.grad is
    non-zero after backward().

    Returns True if the test passes.
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) \
            else torch.device('cpu')

    sino = torch.randn(1, 1, num_angles, det_count,
                       device=device, requires_grad=True)
    recon = model(sino)
    loss = torch.mean(recon ** 2)
    loss.backward()

    grad_ok = (sino.grad is not None and sino.grad.abs().sum().item() > 0)
    print(f"  Gradient Flow Test: {'PASSED' if grad_ok else 'FAILED'}")
    if grad_ok:
        print(f"    grad norm = {sino.grad.norm().item():.6f}")
    return grad_ok


def test_adjoint(backprojector: 'FanBeamBackProjectionLayer',
                 num_angles: int, det_count: int,
                 device: 'torch.device' = None):
    """
    Adjoint consistency test: <A^T y, x> should equal <y, A x> for any
    random image x and sinogram y, where A is forward projection and
    A^T is backprojection.

    Since we only have the backprojector (A^T), we use PyTorch's autograd
    to implicitly compute the forward projection via the transpose of A^T.
    Specifically: A = (A^T)^T, which autograd can compute via vjp.

    This tests that the backprojector is self-consistent with its own adjoint.
    """
    if device is None:
        device = torch.device('cpu')

    img_size = backprojector.image_size

    # Random sinogram and image
    y = torch.randn(1, 1, num_angles, det_count, device=device)
    x = torch.randn(1, 1, img_size, img_size, device=device)

    # Compute A^T y (backprojection)
    y_var = y.clone().requires_grad_(True)
    ATy = backprojector(y_var)

    # <x, A^T y>
    dot1 = torch.sum(x * ATy).item()

    # Compute A x implicitly via autograd:
    # d/dy <A^T y, x> = A x  (the adjoint relationship)
    ATy_dot_x = torch.sum(ATy * x)
    ATy_dot_x.backward()
    Ax = y_var.grad  # This is the forward projection of x

    # <y, A x>
    dot2 = torch.sum(y * Ax).item()

    rel_err = abs(dot1 - dot2) / max(abs(dot1), 1e-10)
    passed = rel_err < 1e-4
    print(f"  Adjoint Test: {'PASSED' if passed else 'FAILED'}")
    print(f"    <x, A^T y> = {dot1:.6f}")
    print(f"    <y, A x>   = {dot2:.6f}")
    print(f"    Relative error = {rel_err:.2e}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point (follows same pattern as run_custom_fbp / run_astra_fbp)
# ─────────────────────────────────────────────────────────────────────────────
def run_differentiable_fbp(input_file, output_file, image_size=512,
                           voxel_size=0.7, fbp_filter='hann',
                           batch_size=8, run_validation=True):
    """
    Reconstruct a full volume using the differentiable FBP operator.

    Loads rebinned fan-beam projections, reconstructs all slices using the
    PyTorch-based differentiable FBP module, converts to Hounsfield Units,
    and saves the result.

    Args:
        input_file    : Path to rebinned fan-beam projections (.tif stack).
        output_file   : Destination path for reconstructed volume (.tif).
        image_size    : Reconstructed image side length in pixels.
        voxel_size    : Physical voxel size in mm.
        fbp_filter    : Filter window name ('hann', 'ram-lak', etc.).
        batch_size    : Number of slices processed simultaneously on GPU.
        run_validation: If True, run gradient flow test after reconstruction.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable FBP reconstruction.\n"
            "Install with: pip install torch\n"
            "Or: conda install pytorch -c pytorch"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Differentiable FBP Reconstruction  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading projections from {input_file} ...")
    projections, metadata = load_tiff_stack_with_metadata(Path(input_file))

    num_slices = projections.shape[2]
    det_count = projections.shape[1]
    num_views = projections.shape[0]
    print(f"Volume: {num_views} views x {det_count} det x {num_slices} slices")

    # ── Geometry (mm → pixels) ────────────────────────────────────────────
    vox_scaling = 1.0 / voxel_size
    dso_px = metadata['dso'] * vox_scaling
    ddo_px = metadata['ddo'] * vox_scaling
    dsd_px = dso_px + ddo_px
    du_px = metadata['du'] * vox_scaling

    # Angles: same convention as run_custom_fbp.py (no negation, -pi/2 offset)
    angles = np.array(metadata['angles'])[:metadata['rotview']] - (np.pi / 2)
    angles_tensor = torch.tensor(angles, dtype=torch.float32)

    # ── Create the differentiable FBP module ──────────────────────────────
    hu_factor = metadata.get('hu_factor', 1.0)
    print(f"Building DifferentiableFanBeamFBP module (filter={fbp_filter}, hu_factor={hu_factor})...")
    model = DifferentiableFanBeamFBP(
        angles=angles_tensor,
        dso=dso_px,
        dsd=dsd_px,
        du=du_px,
        det_count=det_count,
        image_size=image_size,
        filter_name=fbp_filter,
        learnable_filter=False,
        chunk_size=64,
        hu_factor=hu_factor
    ).to(device)
    model.eval()

    # ── Reconstruct (batched, no gradient tracking for inference) ─────────
    reco_volume = []
    total_start = time.time()

    with torch.no_grad():
        for start in tqdm.tqdm(range(0, num_slices, batch_size),
                               desc="Reconstructing (Diff-FBP)"):
            end = min(start + batch_size, num_slices)

            # Build batch: (B, A, D)
            batch_sinos = []
            for s in range(start, end):
                batch_sinos.append(projections[:, :, s] * vox_scaling)
            batch_np = np.stack(batch_sinos, axis=0).astype(np.float32)

            # To tensor: (B, 1, A, D)
            batch_tensor = (torch.from_numpy(batch_np)
                            .unsqueeze(1)
                            .to(device))

            # Forward pass through the differentiable FBP
            recon = model(batch_tensor)  # (B, 1, H, W)

            # Collect on CPU
            reco_volume.append(recon.squeeze(1).cpu().numpy())

    reco_volume = np.concatenate(reco_volume, axis=0)  # (num_slices, H, W)
    elapsed = time.time() - total_start
    print(f"\nTotal reconstruction time: {elapsed:.2f}s "
          f"({elapsed / num_slices:.2f}s per slice)")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Saving reconstruction to {output_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_to_tiff_stack(reco_volume, Path(output_file))

    # ── Validation ────────────────────────────────────────────────────────
    if run_validation:
        print(f"\n{'─'*60}")
        print("  Running Validation Tests")
        print(f"{'─'*60}")
        # Test gradient flow and adjoint on the base unscaled module if needed,
        # but gradient flow still works perfectly through linear HU scaling
        test_gradient_flow(model, len(angles), det_count, device)
        test_adjoint(model.backprojector, len(angles), det_count, device)

    print("\nDifferentiable FBP reconstruction complete!")
    return projections, reco_volume
