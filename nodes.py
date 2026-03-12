"""
YAK — ComfyUI custom nodes for CorridorKey green screen keying
Wraps the CorridorKey neural-network green-screen keying CLI.
https://github.com/nikopueringer/CorridorKey
"""

import os
import sys
import subprocess
import re
import glob as _glob
import tempfile
import json

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_python(ck_dir: str) -> str:
    """Return the venv Python inside a CorridorKey installation, or sys.executable."""
    candidates = [
        os.path.join(ck_dir, ".venv", "Scripts", "python.exe"),  # Windows
        os.path.join(ck_dir, ".venv", "bin", "python"),          # Linux / Mac
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return sys.executable


def _run(cmd: list, cwd: str, timeout: int = 7200) -> tuple[bool, str]:
    """Run a subprocess; return (success, combined_log)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        log = result.stdout + result.stderr
        return result.returncode == 0, log
    except subprocess.TimeoutExpired:
        return False, "ERROR: process timed out."
    except Exception as e:
        return False, f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — CheckSetup
# ─────────────────────────────────────────────────────────────────────────────

class YAKCheckSetup:
    """Verify a CorridorKey installation and report GPU / VRAM status."""

    CATEGORY = "YAK"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("status", "is_ready", "vram_gb", "gpu_name")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "corridorkey_dir": ("STRING", {
                    "default": "C:/CorridorKey",
                    "multiline": False,
                    "tooltip": "Root folder of your CorridorKey installation"
                }),
            }
        }

    def check(self, corridorkey_dir: str):
        ck_dir = corridorkey_dir.rstrip("/\\")

        if not os.path.isdir(ck_dir):
            return ("ERROR: directory not found.", False, 0.0, "Unknown")

        cli = os.path.join(ck_dir, "corridorkey_cli.py")
        if not os.path.isfile(cli):
            return (f"ERROR: corridorkey_cli.py missing in {ck_dir}", False, 0.0, "Unknown")

        python = _find_python(ck_dir)
        vram_gb, gpu_name = 0.0, "Unknown"

        # Try test_vram.py first (bundled with CorridorKey)
        vram_script = os.path.join(ck_dir, "test_vram.py")
        if os.path.isfile(vram_script):
            ok, log = _run([python, vram_script], cwd=ck_dir, timeout=30)
        else:
            # Fallback: query torch directly
            ok, log = _run(
                [python, "-c",
                 "import torch; d=torch.cuda.get_device_properties(0);"
                 "print('GPU:',d.name,'| VRAM:',round(d.total_memory/1073741824,2),'GB')"],
                cwd=ck_dir, timeout=15
            )

        m_gpu  = re.search(r"GPU[:\s]+([^\|\n]+)", log, re.IGNORECASE)
        m_vram = re.search(r"VRAM[:\s]+([\d.]+)\s*GB", log, re.IGNORECASE)
        if m_gpu:
            gpu_name = m_gpu.group(1).strip()
        if m_vram:
            vram_gb = float(m_vram.group(1))

        is_ready = vram_gb >= 6.0
        status = (
            f"Install : OK\n"
            f"Python  : {python}\n"
            f"GPU     : {gpu_name}\n"
            f"VRAM    : {vram_gb:.2f} GB {'(OK)' if is_ready else '(below 6 GB minimum)'}"
        )
        return (status, is_ready, vram_gb, gpu_name)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — ProcessClip
# ─────────────────────────────────────────────────────────────────────────────

class MaskKey_Green:
    """
    Key a single image or video clip with CorridorKey.
    Returns rgb and alpha as IMAGE tensors (PNG-compatible) ready for any ComfyUI node.
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
                    "tooltip": "Connect an image or video frames from any ComfyUI node"
                }),
                "corridorkey_dir": ("STRING", {
                    "default": "/workspace/CorridorKey",
                    "multiline": False,
                    "tooltip": "Root folder of your CorridorKey installation"
                }),
            },
            "optional": {
                "clip_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override: provide a direct path to a video file instead"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output folder (leave empty to use <corridorkey_dir>/Output)"
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "tooltip": "FPS used when saving IMAGE batch as a temp video"
                }),
                "use_gvm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Generative Video Matting"
                }),
                "extra_args": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Extra CLI flags, e.g. --videomama"
                }),
            }
        }

    def _save_frames_as_video(self, images: torch.Tensor, fps: int) -> str:
        """Save IMAGE batch tensor (N,H,W,3) to a temp MP4, return path."""
        tmp_dir = tempfile.mkdtemp(prefix="yak_frames_")
        n = images.shape[0]
        for i in range(n):
            frame = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(tmp_dir, f"frame_{i:05d}.png"))

        tmp_video = tempfile.mktemp(suffix=".mp4", prefix="yak_input_")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp_video
        ]
        subprocess.run(cmd, capture_output=True)
        return tmp_video

    def process(
        self,
        images: torch.Tensor,
        corridorkey_dir: str,
        clip_path: str = "",
        output_dir: str = "",
        fps: int = 24,
        use_gvm: bool = False,
        extra_args: str = "",
    ):
        ck_dir = corridorkey_dir.rstrip("/\\")

        if not os.path.isdir(ck_dir):
            return ("", False, f"ERROR: corridorkey_dir not found: {ck_dir}")

        cli = os.path.join(ck_dir, "corridorkey_cli.py")
        if not os.path.isfile(cli):
            return ("", False, f"ERROR: corridorkey_cli.py missing in {ck_dir}")

        # Resolve input — clip_path overrides images if provided
        if clip_path and os.path.isfile(clip_path):
            pass  # use clip_path as-is
        else:
            clip_path = self._save_frames_as_video(images, fps)

        out_path = output_dir.strip() if output_dir and output_dir.strip() else os.path.join(ck_dir, "Output")
        os.makedirs(out_path, exist_ok=True)

        python = _find_python(ck_dir)
        cmd = [python, cli, "--clip", clip_path, "--output", out_path]

        if use_gvm:
            cmd.append("--gvm")
        if extra_args.strip():
            cmd += extra_args.strip().split()

        success, log = _run(cmd, cwd=ck_dir)

        if not success:
            blank = torch.zeros(1, images.shape[1], images.shape[2], 3)
            mask  = torch.zeros(1, images.shape[1], images.shape[2])
            return (blank, blank, mask, False, log)

        # Load output frames (EXR or PNG/JPG — whatever CorridorKey wrote)
        exr_files = sorted(_glob.glob(os.path.join(out_path, "*.exr")))
        png_files = sorted(_glob.glob(os.path.join(out_path, "*.png")))
        files = exr_files if exr_files else png_files

        rgb_frames, alpha_frames = [], []
        for f in files:
            if f.endswith(".exr"):
                rgba = self._read_exr(f)               # (H,W,4) float32
            else:
                img  = np.array(Image.open(f).convert("RGBA")).astype(np.float32) / 255.0
                rgba = img                              # (H,W,4)

            rgb  = rgba[..., :3]
            alpha = rgba[..., 3:4]

            # Apply sRGB gamma to linear EXR data so it displays correctly
            if f.endswith(".exr"):
                rgb   = np.clip(rgb,   0.0, 1.0) ** (1.0 / 2.2)
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                rgb   = np.clip(rgb,   0.0, 1.0)
                alpha = np.clip(alpha, 0.0, 1.0)

            rgb_frames.append(rgb)
            alpha_frames.append(np.repeat(alpha, 3, axis=-1))

        rgb_t   = torch.from_numpy(np.stack(rgb_frames,   axis=0))   # (N,H,W,3)
        alpha_t = torch.from_numpy(np.stack(alpha_frames, axis=0))   # (N,H,W,3)
        mask_t  = alpha_t[..., 0]                                     # (N,H,W)

        return (rgb_t, alpha_t, mask_t, True, log)

    @staticmethod
    def _read_exr(path: str):
        try:
            import OpenEXR, Imath
            exr = OpenEXR.InputFile(path)
            dw  = exr.header()["dataWindow"]
            W   = dw.max.x - dw.min.x + 1
            H   = dw.max.y - dw.min.y + 1
            pt  = Imath.PixelType(Imath.PixelType.FLOAT)
            ch  = {c: np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(H, W)
                   for c in ("R", "G", "B", "A")}
            return np.stack([ch["R"], ch["G"], ch["B"], ch["A"]], axis=-1)
        except ImportError:
            import imageio
            img = imageio.imread(path, format="exr").astype(np.float32)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[2] == 3:
                img = np.concatenate([img, np.ones((*img.shape[:2], 1), np.float32)], axis=-1)
            return img


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — BatchProcess
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
                "corridorkey_dir": ("STRING", {
                    "default": "C:/CorridorKey",
                    "multiline": False,
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root output folder — a sub-folder is created per clip"
                }),
                "stop_on_error": ("BOOLEAN", {"default": False}),
                "use_gvm": ("BOOLEAN", {"default": False}),
            }
        }

    def batch(
        self,
        clip_paths: str,
        corridorkey_dir: str,
        output_dir: str = "",
        stop_on_error: bool = False,
        use_gvm: bool = False,
    ):
        ck_dir = corridorkey_dir.rstrip("/\\")
        cli    = os.path.join(ck_dir, "corridorkey_cli.py")
        python = _find_python(ck_dir)

        if not os.path.isdir(ck_dir) or not os.path.isfile(cli):
            return ("", f"ERROR: invalid corridorkey_dir: {ck_dir}", False)

        clips = [p.strip() for p in clip_paths.splitlines() if p.strip()]
        if not clips:
            return ("", "ERROR: no clip paths provided.", False)

        folders, logs, successes = [], [], []

        for clip in clips:
            if not os.path.isfile(clip):
                logs.append(f"SKIP (not found): {clip}")
                folders.append("")
                successes.append(False)
                if stop_on_error:
                    break
                continue

            stem     = os.path.splitext(os.path.basename(clip))[0]
            out_root = output_dir.strip() if output_dir and output_dir.strip() else os.path.join(ck_dir, "Output")
            out_path = os.path.join(out_root, stem)
            os.makedirs(out_path, exist_ok=True)

            cmd = [python, cli, "--clip", clip, "--output", out_path]
            if use_gvm:
                cmd.append("--gvm")

            success, log = _run(cmd, cwd=ck_dir)
            folders.append(out_path if success else "")
            logs.append(f"[{stem}]\n{log}")
            successes.append(success)

            if not success and stop_on_error:
                break

        output_folders  = "\n".join(folders)
        combined_log    = "\n\n".join(logs)
        all_succeeded   = bool(successes) and all(successes)
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
        chans = {}
        for ch in ("R", "G", "B", "A"):
            raw = exr.channel(ch, pt)
            chans[ch] = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        rgba = np.stack([chans["R"], chans["G"], chans["B"], chans["A"]], axis=-1)
        return rgba

    @staticmethod
    def _read_exr_imageio(path: str):
        """Read one EXR frame via imageio (freeimage plugin). Returns (H,W,4) float32."""
        import imageio
        img = imageio.imread(path, format="exr")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 3:
            h, w = img.shape[:2]
            alpha = np.ones((h, w, 1), dtype=np.float32)
            img = np.concatenate([img, alpha], axis=-1)
        return img.astype(np.float32)

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

        frames = [self._read_exr(f) for f in files]  # list of (H,W,4)
        batch  = np.stack(frames, axis=0)             # (N,H,W,4)

        rgb_t   = torch.from_numpy(batch[..., :3])    # (N,H,W,3)
        alpha_t = torch.from_numpy(
            np.repeat(batch[..., 3:4], 3, axis=-1)   # broadcast alpha → (N,H,W,3)
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
