"""
YAK — MatAnyone node for video matting with first-frame mask guidance.
Wraps the MatAnyone framework: https://github.com/pq-yang/MatAnyone
"""

import os
import tempfile

import numpy as np
import torch
from PIL import Image


class YAKMatAnyoneGenerate:
    """
    Video matting using MatAnyone (CVPR 2025).
    Takes video frames + an optional first-frame mask, returns per-frame alpha
    mattes, foreground composites, and green-screen composites.
    """

    CATEGORY = "YAK/MatAnyone"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("foreground", "green_screen", "alpha", "mask", "success", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames as IMAGE batch (N,H,W,3)"
                }),
            },
            "optional": {
                "first_frame_mask": ("MASK", {
                    "tooltip": "Binary mask for the target object on the first frame "
                               "(if omitted, uses full-white mask — entire frame is foreground)"
                }),
                "n_warmup": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Number of warmup iterations on first frame to stabilize memory"
                }),
                "r_erode": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Erosion radius for mask preprocessing"
                }),
                "r_dilate": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Dilation radius for mask preprocessing"
                }),
                "max_size": ("INT", {
                    "default": 1280,
                    "min": -1,
                    "max": 4096,
                    "tooltip": "Max dimension for processing (-1 = no resize)"
                }),
                "device": (["auto", "cuda", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for inference"
                }),
            },
        }

    def generate(
        self,
        images: torch.Tensor,
        first_frame_mask: torch.Tensor = None,
        n_warmup: int = 10,
        r_erode: int = 10,
        r_dilate: int = 10,
        max_size: int = 1280,
        device: str = "auto",
    ):
        h, w = images.shape[1], images.shape[2]
        blank = torch.zeros(1, h, w, 3)
        blank_green = torch.zeros(1, h, w, 3)
        blank_green[..., 1] = 1.0
        mask_blank = torch.zeros(1, h, w)

        # Default to full-white mask if none provided
        if first_frame_mask is None:
            first_frame_mask = torch.ones(1, h, w)

        try:
            from matanyone import InferenceCore
            from matanyone.utils.device import safe_autocast
            from matanyone.utils.inference_utils import gen_dilate, gen_erosion
        except ImportError:
            return (blank, blank_green, blank, mask_blank, False,
                    "ERROR: matanyone package not found.\n"
                    "Install ComfyUI-YAK requirements: pip install -r requirements.txt")

        log_lines = []

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)
        log_lines.append(f"Device: {dev}")

        # Initialize model
        try:
            processor = InferenceCore("PeiqingYang/MatAnyone", device=dev)
        except TypeError:
            processor = InferenceCore("PeiqingYang/MatAnyone")
        log_lines.append("Model loaded from HuggingFace hub")

        n_frames = images.shape[0]

        # Prepare mask — convert to 0-255 numpy for erosion/dilation
        if first_frame_mask.dim() == 3:
            mask_np = first_frame_mask[0].cpu().numpy()
        else:
            mask_np = first_frame_mask.cpu().numpy()
        mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)

        # Apply erosion/dilation like reference process_video
        if r_dilate > 0:
            mask_np = gen_dilate(mask_np, r_dilate, r_dilate)
        if r_erode > 0:
            mask_np = gen_erosion(mask_np, r_erode, r_erode)

        mask_tensor = torch.from_numpy(mask_np).float().to(dev)

        # Convert all frames: ComfyUI (N,H,W,3) float 0-1 → (N,3,H,W) float 0-1
        frames = images.permute(0, 3, 1, 2).float()  # (N,3,H,W)

        # Resize if needed (match reference: resize by min_side)
        resize_needed = False
        if max_size > 0:
            min_side = min(h, w)
            if min_side > max_size:
                resize_needed = True
                new_h = int(h / min_side * max_size)
                new_w = int(w / min_side * max_size)
                frames = torch.nn.functional.interpolate(
                    frames, size=(new_h, new_w), mode="area"
                )
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w), mode="nearest"
                )[0, 0]
                log_lines.append(f"Resized from {w}x{h} to {new_w}x{new_h}")

        # Prepend warmup frames (copies of frame 0) like reference
        if n_warmup > 0:
            warmup_frames = frames[0:1].repeat(n_warmup, 1, 1, 1)
            all_frames = torch.cat([warmup_frames, frames], dim=0)
        else:
            all_frames = frames
        total_len = all_frames.shape[0]

        green_bg = np.array([0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 1, 3)

        alpha_list = []
        fgr_list = []
        green_list = []

        try:
            with torch.inference_mode(), safe_autocast():
                for ti in range(total_len):
                    frame = all_frames[ti].to(dev)  # (3,H,W) float 0-1

                    if ti == 0:
                        # Encode mask on first frame
                        output_prob = processor.step(frame, mask_tensor, objects=[1])
                        output_prob = processor.step(frame, first_frame_pred=True)
                    elif ti <= n_warmup:
                        # Warmup: repeat first frame with first_frame_pred
                        output_prob = processor.step(frame, first_frame_pred=True)
                    else:
                        # Normal frame
                        output_prob = processor.step(frame)

                    # Skip warmup frames from output
                    if ti < n_warmup:
                        continue

                    alpha = processor.output_prob_to_mask(output_prob, matting=True)
                    pha_np = alpha.cpu().numpy()  # (H,W)

                    # Get original frame as numpy (H,W,3) 0-1
                    frame_np = all_frames[ti].permute(1, 2, 0).cpu().numpy()

                    # Foreground composite (over black)
                    pha_3ch = pha_np[:, :, np.newaxis]
                    fgr_np = frame_np * pha_3ch

                    # Green screen composite
                    green_np = frame_np * pha_3ch + green_bg * (1.0 - pha_3ch)

                    alpha_list.append(pha_np)
                    fgr_list.append(fgr_np)
                    green_list.append(green_np)

                    idx = ti - n_warmup
                    if (idx + 1) % 50 == 0:
                        log_lines.append(f"Processed frame {idx + 1}/{n_frames}")

                processor.clear_memory()
                log_lines.append(f"Processing complete: {n_frames} frames")

        except Exception as e:
            log_lines.append(f"ERROR during inference: {e}")
            import traceback
            log_lines.append(traceback.format_exc())
            return (blank, blank_green, blank, mask_blank, False, "\n".join(log_lines))

        # Stack results
        alpha_stack = np.stack(alpha_list, axis=0)  # (N,H',W')
        fgr_stack = np.stack(fgr_list, axis=0)      # (N,H',W',3)
        green_stack = np.stack(green_list, axis=0)   # (N,H',W',3)

        # Resize back to original if needed
        if resize_needed:
            alpha_t = torch.from_numpy(alpha_stack).unsqueeze(1).float()
            alpha_t = torch.nn.functional.interpolate(
                alpha_t, size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
            alpha_stack = alpha_t.numpy()

            fgr_t = torch.from_numpy(fgr_stack).permute(0, 3, 1, 2).float()
            fgr_t = torch.nn.functional.interpolate(
                fgr_t, size=(h, w), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            fgr_stack = fgr_t.numpy()

            green_t = torch.from_numpy(green_stack).permute(0, 3, 1, 2).float()
            green_t = torch.nn.functional.interpolate(
                green_t, size=(h, w), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            green_stack = green_t.numpy()

        # Convert to ComfyUI tensors
        fgr_out = torch.from_numpy(fgr_stack).float().clamp(0, 1)
        green_out = torch.from_numpy(green_stack).float().clamp(0, 1)
        alpha_3ch = np.repeat(np.expand_dims(alpha_stack, -1), 3, axis=-1)
        alpha_out = torch.from_numpy(alpha_3ch).float().clamp(0, 1)
        mask_out = torch.from_numpy(alpha_stack).float().clamp(0, 1)

        return (fgr_out, green_out, alpha_out, mask_out, True, "\n".join(log_lines))


class YAKMatAnyoneCheckSetup:
    """Verify MatAnyone installation and report GPU status."""

    CATEGORY = "YAK/MatAnyone"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "is_ready")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    def check(self):
        lines = []

        try:
            import matanyone  # noqa: F401
            lines.append("Package     : installed")
            pkg_ready = True
        except ImportError:
            lines.append("Package     : NOT installed (pip install -r requirements.txt)")
            pkg_ready = False

        # GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = round(props.total_mem / 1073741824, 2)
            lines.append(f"GPU         : {gpu_name}")
            lines.append(f"VRAM        : {vram_gb} GB")
            gpu_ready = vram_gb >= 4.0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            lines.append("GPU         : Apple MPS")
            gpu_ready = True
        else:
            lines.append("GPU         : CPU only (slow)")
            gpu_ready = True

        is_ready = pkg_ready and gpu_ready
        status = "\n".join(lines)
        return (status, is_ready)
