"""
YAK — MatAnyone node for video matting with first-frame mask guidance.
Wraps the MatAnyone framework: https://github.com/pq-yang/MatAnyone
"""

import os
import sys
import subprocess
import tempfile
import glob as _glob

import numpy as np
import torch
from PIL import Image


def _find_matanyone_python(matanyone_dir: str) -> str:
    """Return the conda/venv Python inside a MatAnyone installation."""
    candidates = [
        os.path.join(matanyone_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(matanyone_dir, ".venv", "bin", "python"),
        os.path.join(matanyone_dir, "venv", "Scripts", "python.exe"),
        os.path.join(matanyone_dir, "venv", "bin", "python"),
        os.path.join(matanyone_dir, "conda_env", "python.exe"),
        os.path.join(matanyone_dir, "conda_env", "bin", "python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return sys.executable


class YAKMatAnyoneGenerate:
    """
    Video matting using MatAnyone (CVPR 2025).
    Takes video frames + a first-frame mask, returns per-frame alpha mattes
    and foreground composites.
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
                "first_frame_mask": ("MASK", {
                    "tooltip": "Binary mask for the target object on the first frame"
                }),
            },
            "optional": {
                "matanyone_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root folder of your MatAnyone installation"
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
                "use_python_api": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Python API (faster, in-process) vs CLI subprocess"
                }),
            },
        }

    def _run_python_api(
        self,
        images: torch.Tensor,
        first_frame_mask: torch.Tensor,
        matanyone_dir: str,
        n_warmup: int,
        r_erode: int,
        r_dilate: int,
        max_size: int,
        device: str,
    ):
        """Run MatAnyone via Python API (in-process, faster)."""
        # Add matanyone to path if needed
        if matanyone_dir and os.path.isdir(matanyone_dir):
            if matanyone_dir not in sys.path:
                sys.path.insert(0, matanyone_dir)

        try:
            from matanyone import InferenceCore
        except ImportError:
            return None, None, False, (
                "ERROR: matanyone package not found. "
                "Install it with: pip install -e . (from the MatAnyone repo) "
                "or provide the matanyone_dir path."
            )

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
            # Older API may not accept device kwarg
            processor = InferenceCore("PeiqingYang/MatAnyone")
        log_lines.append("Model loaded from HuggingFace hub")

        n_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        # Prepare mask: MASK is (N,H,W) or (H,W), we need first frame
        if first_frame_mask.dim() == 3:
            mask_np = first_frame_mask[0].cpu().numpy()
        else:
            mask_np = first_frame_mask.cpu().numpy()
        # Scale to 0-255 range for MatAnyone
        mask_255 = (mask_np * 255).clip(0, 255).astype(np.uint8)

        # Resize if needed
        if max_size > 0 and max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            log_lines.append(f"Resized from {w}x{h} to {new_w}x{new_h}")
        else:
            new_h, new_w = h, w

        # Process frames
        alpha_frames = []
        fgr_frames = []

        try:
            # Convert first frame: (H,W,3) -> (3,H,W) float [0,1]
            frame0 = images[0].cpu().permute(2, 0, 1).float()
            if max_size > 0 and max(h, w) > max_size:
                frame0 = torch.nn.functional.interpolate(
                    frame0.unsqueeze(0), size=(new_h, new_w), mode="bilinear"
                ).squeeze(0)
                mask_pil = Image.fromarray(mask_255).resize((new_w, new_h), Image.NEAREST)
                mask_tensor = torch.from_numpy(np.array(mask_pil)).float()
            else:
                mask_tensor = torch.from_numpy(mask_255).float()

            frame0 = frame0.to(dev)
            mask_tensor = mask_tensor.to(dev)

            # Encode mask on first frame
            processor.step(frame0, mask_tensor, objects=[1])
            output_prob = processor.step(frame0, first_frame_pred=True)

            # Warmup
            for _ in range(n_warmup):
                output_prob = processor.step(frame0, first_frame_pred=True)
            log_lines.append(f"Warmup complete ({n_warmup} iterations)")

            # First frame alpha
            alpha = processor.output_prob_to_mask(output_prob, matting=True)
            alpha_frames.append(alpha.cpu().numpy())

            # Process remaining frames
            for i in range(1, n_frames):
                frame_i = images[i].cpu().permute(2, 0, 1).float()
                if max_size > 0 and max(h, w) > max_size:
                    frame_i = torch.nn.functional.interpolate(
                        frame_i.unsqueeze(0), size=(new_h, new_w), mode="bilinear"
                    ).squeeze(0)
                frame_i = frame_i.to(dev)

                output_prob = processor.step(frame_i)
                alpha = processor.output_prob_to_mask(output_prob, matting=True)
                alpha_frames.append(alpha.cpu().numpy())

                if (i + 1) % 50 == 0:
                    log_lines.append(f"Processed frame {i + 1}/{n_frames}")

            processor.clear_memory()
            log_lines.append(f"Processing complete: {n_frames} frames")

        except Exception as e:
            log_lines.append(f"ERROR during inference: {e}")
            return None, None, False, "\n".join(log_lines)

        # Stack alpha frames and resize back if needed
        alpha_stack = np.stack(alpha_frames, axis=0)  # (N, new_h, new_w)
        if new_h != h or new_w != w:
            alpha_t = torch.from_numpy(alpha_stack).unsqueeze(1).float()
            alpha_t = torch.nn.functional.interpolate(
                alpha_t, size=(h, w), mode="bilinear"
            ).squeeze(1)
            alpha_stack = alpha_t.numpy()

        # Build foreground composites
        images_np = images.cpu().numpy()  # (N,H,W,3)
        alpha_3ch = np.expand_dims(alpha_stack, axis=-1)  # (N,H,W,1)
        fgr_np = images_np * alpha_3ch  # (N,H,W,3)

        return fgr_np, alpha_stack, True, "\n".join(log_lines)

    def _run_cli(
        self,
        images: torch.Tensor,
        first_frame_mask: torch.Tensor,
        matanyone_dir: str,
        n_warmup: int,
        r_erode: int,
        r_dilate: int,
        max_size: int,
    ):
        """Run MatAnyone via CLI subprocess."""
        if not matanyone_dir or not os.path.isdir(matanyone_dir):
            return None, None, False, (
                "ERROR: matanyone_dir is required for CLI mode. "
                "Provide the root folder of your MatAnyone installation."
            )

        # Save frames as video
        tmp_dir = tempfile.mkdtemp(prefix="yak_matanyone_")
        n_frames = images.shape[0]
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i in range(n_frames):
            frame = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:05d}.png"))

        # Save first-frame mask
        if first_frame_mask.dim() == 3:
            mask_np = first_frame_mask[0].cpu().numpy()
        else:
            mask_np = first_frame_mask.cpu().numpy()
        mask_img = (mask_np * 255).clip(0, 255).astype(np.uint8)
        mask_path = os.path.join(tmp_dir, "mask.png")
        Image.fromarray(mask_img, mode="L").save(mask_path)

        out_path = os.path.join(tmp_dir, "output")
        os.makedirs(out_path, exist_ok=True)

        python = _find_matanyone_python(matanyone_dir)
        script = os.path.join(matanyone_dir, "inference_matanyone.py")

        if not os.path.isfile(script):
            return None, None, False, (
                f"ERROR: inference_matanyone.py not found in {matanyone_dir}"
            )

        cmd = [
            python, script,
            "-i", frames_dir,
            "-m", mask_path,
            "-o", out_path,
            "--n_warmup", str(n_warmup),
            "--r_erode", str(r_erode),
            "--r_dilate", str(r_dilate),
            "--save_image",
        ]
        if max_size > 0:
            cmd += ["--max_size", str(max_size)]

        try:
            result = subprocess.run(
                cmd, cwd=matanyone_dir, capture_output=True, text=True, timeout=1800
            )
            log = result.stdout + result.stderr
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            return None, None, False, "ERROR: MatAnyone timed out after 30 minutes."
        except Exception as e:
            return None, None, False, f"ERROR: {e}"

        if not success:
            return None, None, False, log

        # Find output alpha frames
        pha_dir = _glob.glob(os.path.join(out_path, "**", "pha"), recursive=True)
        if not pha_dir:
            # Try finding alpha frames directly
            alpha_files = sorted(_glob.glob(os.path.join(out_path, "**", "*.png"), recursive=True))
            alpha_files = [f for f in alpha_files if "pha" in f.lower() or "alpha" in f.lower()]
        else:
            alpha_files = sorted(_glob.glob(os.path.join(pha_dir[0], "*.png")))

        if not alpha_files:
            return None, None, False, log + "\nERROR: no alpha frames found in output."

        # Load alpha frames
        h, w = images.shape[1], images.shape[2]
        alpha_frames = []
        for f in alpha_files:
            a = np.array(Image.open(f).convert("L")).astype(np.float32) / 255.0
            # Resize back to original if needed
            if a.shape[0] != h or a.shape[1] != w:
                a_pil = Image.fromarray((a * 255).astype(np.uint8), mode="L")
                a_pil = a_pil.resize((w, h), Image.BILINEAR)
                a = np.array(a_pil).astype(np.float32) / 255.0
            alpha_frames.append(a)

        alpha_stack = np.stack(alpha_frames, axis=0)  # (N,H,W)

        # Build foreground composites
        images_np = images.cpu().numpy()[:len(alpha_frames)]
        alpha_3ch = np.expand_dims(alpha_stack, axis=-1)
        fgr_np = images_np * alpha_3ch

        return fgr_np, alpha_stack, True, log

    def generate(
        self,
        images: torch.Tensor,
        first_frame_mask: torch.Tensor,
        matanyone_dir: str = "",
        n_warmup: int = 10,
        r_erode: int = 10,
        r_dilate: int = 10,
        max_size: int = 1280,
        device: str = "auto",
        use_python_api: bool = True,
    ):
        blank = torch.zeros(1, images.shape[1], images.shape[2], 3)
        mask_blank = torch.zeros(1, images.shape[1], images.shape[2])
        blank_green = blank.clone()

        if use_python_api:
            fgr_np, alpha_np, success, log = self._run_python_api(
                images, first_frame_mask, matanyone_dir,
                n_warmup, r_erode, r_dilate, max_size, device,
            )
        else:
            fgr_np, alpha_np, success, log = self._run_cli(
                images, first_frame_mask, matanyone_dir,
                n_warmup, r_erode, r_dilate, max_size,
            )

        if not success or fgr_np is None:
            return (blank, blank_green, blank, mask_blank, False, log)

        fgr_t = torch.from_numpy(fgr_np).float().clamp(0, 1)       # (N,H,W,3)
        alpha_3ch = np.repeat(np.expand_dims(alpha_np, -1), 3, axis=-1)
        alpha_t = torch.from_numpy(alpha_3ch).float().clamp(0, 1)   # (N,H,W,3)
        mask_t = torch.from_numpy(alpha_np).float().clamp(0, 1)     # (N,H,W)

        # Green screen composite: foreground over green background
        green_bg = np.zeros_like(fgr_np)
        green_bg[..., 1] = 1.0  # pure green
        alpha_1ch = np.expand_dims(alpha_np, -1)  # (N,H,W,1)
        green_np = fgr_np + green_bg * (1.0 - alpha_1ch)
        green_t = torch.from_numpy(green_np).float().clamp(0, 1)

        return (fgr_t, green_t, alpha_t, mask_t, success, log)


class YAKMatAnyoneCheckSetup:
    """Verify a MatAnyone installation and report GPU status."""

    CATEGORY = "YAK/MatAnyone"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "is_ready")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matanyone_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root folder of your MatAnyone installation (leave empty to check pip install)"
                }),
            }
        }

    def check(self, matanyone_dir: str = ""):
        lines = []

        # Check if matanyone is importable
        if matanyone_dir and os.path.isdir(matanyone_dir):
            if matanyone_dir not in sys.path:
                sys.path.insert(0, matanyone_dir)
            lines.append(f"MatAnyone dir: {matanyone_dir}")

            script = os.path.join(matanyone_dir, "inference_matanyone.py")
            if os.path.isfile(script):
                lines.append("CLI script  : found")
            else:
                lines.append("CLI script  : NOT FOUND")

        try:
            import matanyone  # noqa: F401
            lines.append("Package     : installed")
            pkg_ready = True
        except ImportError:
            lines.append("Package     : NOT installed (pip install -e . from MatAnyone repo)")
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
            gpu_ready = True  # CPU works, just slow

        is_ready = pkg_ready and gpu_ready
        status = "\n".join(lines)
        return (status, is_ready)
