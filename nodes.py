"""
YAK — ComfyUI custom nodes for CorridorKey green screen keying
Uses the CorridorKey Python API directly (pip-installed).
https://github.com/nikopueringer/CorridorKey
"""

import os
import glob as _glob
import json

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint auto-download
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
CHECKPOINT_NAME = "CorridorKey_v1.0.pth"


def _get_checkpoint() -> str:
    """Return path to the CorridorKey checkpoint, downloading if needed."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    if os.path.isfile(ckpt_path):
        return ckpt_path

    # Try huggingface_hub first (resumable, cached)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="nikopueringer/CorridorKey_v1.0",
            filename="CorridorKey_v1.0.pth",
            local_dir=CHECKPOINT_DIR,
        )
        return downloaded
    except Exception:
        pass

    # Fallback: urllib
    import urllib.request
    print(f"[YAK] Downloading CorridorKey checkpoint to {ckpt_path}...")
    urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — CheckSetup
# ─────────────────────────────────────────────────────────────────────────────

class YAKCheckSetup:
    """Verify CorridorKey installation and report GPU / VRAM status."""

    CATEGORY = "YAK"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("status", "is_ready", "vram_gb", "gpu_name")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def check(self):
        lines = []

        # Check CorridorKey module
        try:
            from CorridorKeyModule import CorridorKeyEngine  # noqa: F401
            lines.append("Package : installed")
            pkg_ready = True
        except ImportError:
            lines.append("Package : NOT installed (pip install -r requirements.txt)")
            pkg_ready = False

        # Check checkpoint
        ckpt = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        if os.path.isfile(ckpt):
            lines.append(f"Checkpoint: found")
        else:
            lines.append("Checkpoint: not downloaded (will auto-download on first run)")

        # GPU info
        vram_gb = 0.0
        gpu_name = "Unknown"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = round(props.total_mem / 1073741824, 2)
            lines.append(f"GPU     : {gpu_name}")
            lines.append(f"VRAM    : {vram_gb:.2f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_name = "Apple MPS"
            lines.append("GPU     : Apple MPS")
        else:
            lines.append("GPU     : CPU only (slow)")

        is_ready = pkg_ready and vram_gb >= 6.0
        status = "\n".join(lines)
        return (status, is_ready, vram_gb, gpu_name)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — MaskKey_Green
# ─────────────────────────────────────────────────────────────────────────────

class MaskKey_Green:
    """
    Key green screen footage with CorridorKey.
    Uses the Python API directly — no CLI or manual installation needed.
    Auto-downloads the model checkpoint on first run.
    """

    CATEGORY = "YAK"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("rgb", "alpha", "mask", "success", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Green screen footage — image or video frames"
                }),
            },
            "optional": {
                "alpha_hint": ("MASK", {
                    "tooltip": "Optional rough alpha hint mask (auto-generated if not connected)"
                }),
                "background": (["green", "blue", "red", "white", "black", "checkerboard"], {
                    "default": "checkerboard",
                    "tooltip": "Background for the composited foreground output"
                }),
                "despill_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Green spill removal strength"
                }),
                "auto_despeckle": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected alpha islands"
                }),
                "device": (["auto", "cuda", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for inference"
                }),
            }
        }

    def process(
        self,
        images: torch.Tensor,
        alpha_hint: torch.Tensor = None,
        background: str = "checkerboard",
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        device: str = "auto",
    ):
        h, w = images.shape[1], images.shape[2]
        n_frames = images.shape[0]
        blank = torch.zeros(1, h, w, 3)
        mask_blank = torch.zeros(1, h, w)

        try:
            from CorridorKeyModule import CorridorKeyEngine
        except ImportError:
            return (blank, blank, mask_blank, False,
                    "ERROR: CorridorKeyModule not found.\n"
                    "Install with: pip install -r requirements.txt")

        log_lines = []

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device
        log_lines.append(f"Device: {dev}")

        # Download checkpoint if needed
        try:
            ckpt = _get_checkpoint()
            log_lines.append(f"Checkpoint: {os.path.basename(ckpt)}")
        except Exception as e:
            return (blank, blank, mask_blank, False,
                    f"ERROR: Failed to download checkpoint: {e}")

        # Initialize engine
        try:
            engine = CorridorKeyEngine(
                checkpoint_path=ckpt,
                device=dev,
                img_size=2048,
            )
            log_lines.append("CorridorKey engine loaded")
        except Exception as e:
            log_lines.append(f"ERROR initializing engine: {e}")
            import traceback
            log_lines.append(traceback.format_exc())
            return (blank, blank, mask_blank, False, "\n".join(log_lines))

        # Generate alpha hint if not provided (simple green detection)
        if alpha_hint is not None:
            if alpha_hint.dim() == 3:
                hint_np = alpha_hint[0].cpu().numpy()
            else:
                hint_np = alpha_hint.cpu().numpy()
        else:
            hint_np = None

        rgb_frames = []
        alpha_frames = []

        try:
            for i in range(n_frames):
                frame_np = images[i].cpu().numpy()  # (H,W,3) float 0-1

                # Per-frame alpha hint
                if hint_np is not None:
                    mask_in = hint_np
                elif alpha_hint is not None and alpha_hint.dim() == 3 and i < alpha_hint.shape[0]:
                    mask_in = alpha_hint[i].cpu().numpy()
                else:
                    # Auto-generate rough alpha hint from green channel
                    mask_in = self._green_hint(frame_np)

                result = engine.process_frame(
                    image=frame_np,
                    mask_linear=mask_in,
                    despill_strength=despill_strength,
                    auto_despeckle=auto_despeckle,
                )

                alpha_out = result["alpha"]  # (H,W,1) linear
                fg_out = result["fg"]        # (H,W,3) sRGB straight

                if alpha_out.ndim == 3:
                    alpha_out = alpha_out[:, :, 0]  # (H,W)

                # Composite foreground over chosen background
                a = alpha_out[:, :, np.newaxis]  # (H,W,1)
                if background == "checkerboard":
                    comp = result["comp"]  # Already composited on checkerboard
                else:
                    bg_colors = {
                        "green": [0.0, 1.0, 0.0],
                        "blue":  [0.0, 0.0, 1.0],
                        "red":   [1.0, 0.0, 0.0],
                        "white": [1.0, 1.0, 1.0],
                        "black": [0.0, 0.0, 0.0],
                    }
                    bg = np.array(bg_colors.get(background, [0, 0, 0]),
                                  dtype=np.float32).reshape(1, 1, 3)
                    comp = fg_out * a + bg * (1.0 - a)

                rgb_frames.append(np.clip(comp, 0, 1))
                alpha_frames.append(np.clip(alpha_out, 0, 1))

                if (i + 1) % 50 == 0:
                    log_lines.append(f"Processed frame {i + 1}/{n_frames}")

            log_lines.append(f"Processing complete: {n_frames} frames")

        except Exception as e:
            log_lines.append(f"ERROR during inference: {e}")
            import traceback
            log_lines.append(traceback.format_exc())
            return (blank, blank, mask_blank, False, "\n".join(log_lines))

        rgb_stack = np.stack(rgb_frames, axis=0)      # (N,H,W,3)
        alpha_stack = np.stack(alpha_frames, axis=0)   # (N,H,W)

        rgb_t = torch.from_numpy(rgb_stack).float()
        alpha_3ch = np.repeat(np.expand_dims(alpha_stack, -1), 3, axis=-1)
        alpha_t = torch.from_numpy(alpha_3ch).float()
        mask_t = torch.from_numpy(alpha_stack).float()

        return (rgb_t, alpha_t, mask_t, True, "\n".join(log_lines))

    @staticmethod
    def _green_hint(frame: np.ndarray) -> np.ndarray:
        """Generate a rough alpha hint from green channel dominance."""
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        # Green is dominant and bright enough
        green_mask = (g > r * 1.2) & (g > b * 1.2) & (g > 0.15)
        # Invert: foreground = non-green areas = 1, green = 0
        hint = (~green_mask).astype(np.float32)
        return hint


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — BatchProcess (kept for compatibility, uses Python API)
# ─────────────────────────────────────────────────────────────────────────────

class YAKBatchProcess:
    """
    Key multiple video clips in sequence.
    Clip paths are provided as a newline-separated string.
    """

    CATEGORY = "YAK"
    FUNCTION = "batch"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("output_folders", "combined_log", "all_succeeded")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One clip path per line"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root output folder — a sub-folder is created per clip"
                }),
                "stop_on_error": ("BOOLEAN", {"default": False}),
            }
        }

    def batch(
        self,
        clip_paths: str,
        output_dir: str = "",
        stop_on_error: bool = False,
    ):
        clips = [p.strip() for p in clip_paths.splitlines() if p.strip()]
        if not clips:
            return ("", "ERROR: no clip paths provided.", False)

        try:
            from CorridorKeyModule import CorridorKeyEngine
        except ImportError:
            return ("", "ERROR: CorridorKeyModule not found.", False)

        try:
            ckpt = _get_checkpoint()
        except Exception as e:
            return ("", f"ERROR: checkpoint download failed: {e}", False)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        engine = CorridorKeyEngine(checkpoint_path=ckpt, device=dev, img_size=2048)

        folders, logs, successes = [], [], []

        for clip in clips:
            if not os.path.isfile(clip):
                logs.append(f"SKIP (not found): {clip}")
                folders.append("")
                successes.append(False)
                if stop_on_error:
                    break
                continue

            stem = os.path.splitext(os.path.basename(clip))[0]
            out_root = output_dir.strip() if output_dir and output_dir.strip() else os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Output"
            )
            out_path = os.path.join(out_root, stem)
            os.makedirs(out_path, exist_ok=True)

            try:
                import cv2
                cap = cv2.VideoCapture(clip)
                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    hint = MaskKey_Green._green_hint(frame_rgb)
                    result = engine.process_frame(image=frame_rgb, mask_linear=hint)
                    # Save processed RGBA as EXR or PNG
                    alpha = result["alpha"]
                    if alpha.ndim == 3:
                        alpha = alpha[:, :, 0]
                    fg = result["fg"]
                    rgba = np.concatenate([fg, alpha[:, :, np.newaxis]], axis=-1)
                    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(rgba_uint8, mode="RGBA").save(
                        os.path.join(out_path, f"{idx:05d}.png")
                    )
                    idx += 1
                cap.release()
                logs.append(f"[{stem}] Processed {idx} frames")
                folders.append(out_path)
                successes.append(True)
            except Exception as e:
                logs.append(f"[{stem}] ERROR: {e}")
                folders.append("")
                successes.append(False)
                if stop_on_error:
                    break

        output_folders = "\n".join(folders)
        combined_log = "\n\n".join(logs)
        all_succeeded = bool(successes) and all(successes)
        return (output_folders, combined_log, all_succeeded)


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — LoadEXRSequence
# ─────────────────────────────────────────────────────────────────────────────

class YAKLoadEXRSequence:
    """
    Load a folder of EXR frames output by CorridorKey into a ComfyUI IMAGE batch tensor.
    Requires OpenEXR + Imath  (pip install openexr imath)  OR  imageio[freeimage].
    Falls back to imageio if the openexr package is unavailable.
    """

    CATEGORY = "YAK"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("rgb", "alpha", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder containing the .exr frames from CorridorKey"
                }),
            },
            "optional": {
                "frame_limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "Max frames to load (0 = all)"
                }),
            }
        }

    @staticmethod
    def _read_exr_openexr(path: str):
        """Read one EXR frame via the openexr package. Returns (H,W,4) float32 RGBA."""
        import OpenEXR
        import Imath
        exr  = OpenEXR.InputFile(path)
        dw   = exr.header()["dataWindow"]
        W    = dw.max.x - dw.min.x + 1
        H    = dw.max.y - dw.min.y + 1
        pt   = Imath.PixelType(Imath.PixelType.FLOAT)

        available = list(exr.header()["channels"].keys())
        zeros = np.zeros((H, W), dtype=np.float32)
        ones  = np.ones((H, W), dtype=np.float32)

        def read_ch(name, default):
            if name in available:
                return np.frombuffer(exr.channel(name, pt), dtype=np.float32).reshape(H, W)
            return default

        r = read_ch("R", zeros)
        g = read_ch("G", zeros)
        b = read_ch("B", zeros)
        a = read_ch("A", ones)

        if "Y" in available and "R" not in available:
            y = np.frombuffer(exr.channel("Y", pt), dtype=np.float32).reshape(H, W)
            a = y
            r = g = b = zeros

        return np.stack([r, g, b, a], axis=-1)

    @staticmethod
    def _read_exr_imageio(path: str):
        """Read one EXR frame via imageio (freeimage plugin). Returns (H,W,4) float32."""
        import imageio
        try:
            img = imageio.imread(path, plugin="freeimage").astype(np.float32)
        except Exception:
            img = imageio.imread(path, format="exr").astype(np.float32)
        if img.ndim == 2:
            alpha = img
            img = np.stack([np.zeros_like(alpha)] * 3 + [alpha], axis=-1)
        elif img.shape[2] == 3:
            h, w = img.shape[:2]
            alpha = np.ones((h, w, 1), dtype=np.float32)
            img = np.concatenate([img, alpha], axis=-1)
        return img

    def _read_exr(self, path: str):
        try:
            return self._read_exr_openexr(path)
        except ImportError:
            return self._read_exr_imageio(path)

    def load(self, exr_folder: str, frame_limit: int = 0):
        files = sorted(_glob.glob(os.path.join(exr_folder, "*.exr")))
        if not files:
            raise FileNotFoundError(f"No .exr files found in: {exr_folder}")

        if frame_limit and frame_limit > 0:
            files = files[:frame_limit]

        frames = [self._read_exr(f) for f in files]
        batch  = np.stack(frames, axis=0)

        rgb_t   = torch.from_numpy(batch[..., :3])
        alpha_t = torch.from_numpy(
            np.repeat(batch[..., 3:4], 3, axis=-1)
        )
        return (rgb_t, alpha_t, len(files))


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — 3D Viewport
# ─────────────────────────────────────────────────────────────────────────────

MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx"}
SPLAT_EXTENSIONS = {".ply", ".splat", ".spz"}


def _validate_path(path: str, allowed_exts: set) -> str | None:
    """Return normalised path if valid and has an allowed extension, else None."""
    if not path or not path.strip():
        return None
    path = os.path.normpath(path.strip())
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_exts:
        return None
    return path


class YAK3DViewport:
    """
    Interactive 3D viewport that displays up to 3 meshes and 3 Gaussian splats.
    Supports .glb/.gltf/.obj/.fbx meshes and .ply/.splat/.spz splats.
    """

    CATEGORY = "YAK/3D"
    FUNCTION = "view"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        mesh_opts = {"default": "", "multiline": False, "tooltip": "Path to .glb/.gltf/.obj/.fbx"}
        splat_opts = {"default": "", "multiline": False, "tooltip": "Path to .ply/.splat/.spz"}
        return {
            "required": {},
            "optional": {
                "mesh_1": ("STRING", {**mesh_opts}),
                "mesh_2": ("STRING", {**mesh_opts}),
                "mesh_3": ("STRING", {**mesh_opts}),
                "splat_1": ("STRING", {**splat_opts}),
                "splat_2": ("STRING", {**splat_opts}),
                "splat_3": ("STRING", {**splat_opts}),
                "bg_color": ("STRING", {
                    "default": "#1a1a1a",
                    "multiline": False,
                    "tooltip": "Viewport background colour (hex)"
                }),
                "grid": ("BOOLEAN", {"default": True, "tooltip": "Show ground grid"}),
            },
        }

    def view(
        self,
        mesh_1="", mesh_2="", mesh_3="",
        splat_1="", splat_2="", splat_3="",
        bg_color="#1a1a1a",
        grid=True,
    ):
        meshes = []
        for p in (mesh_1, mesh_2, mesh_3):
            valid = _validate_path(p, MESH_EXTENSIONS)
            if valid:
                meshes.append(valid)

        splats = []
        for p in (splat_1, splat_2, splat_3):
            valid = _validate_path(p, SPLAT_EXTENSIONS)
            if valid:
                splats.append(valid)

        viewport_data = {
            "meshes": meshes,
            "splats": splats,
            "bg_color": bg_color,
            "grid": grid,
        }

        return {"ui": {"viewport_data": [json.dumps(viewport_data)]}, "result": ()}
