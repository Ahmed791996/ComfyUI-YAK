"""
YAK — World Labs API nodes for generating 3D worlds from text/image/video/multi-view.
API docs: https://docs.worldlabs.ai
"""

import os
import json
import time
import tempfile
import mimetypes

import numpy as np
import torch
from PIL import Image

try:
    import requests
except ImportError:
    requests = None

WL_API_BASE = "https://api.worldlabs.ai"


def _require_requests():
    if requests is None:
        raise ImportError(
            "The 'requests' library is required for World Labs nodes. "
            "Install it with: pip install requests"
        )


def _wl_headers(api_key: str) -> dict:
    return {
        "WLT-Api-Key": api_key,
        "Content-Type": "application/json",
    }


def _upload_media(api_key: str, filepath: str) -> str:
    """Upload a file to World Labs, return media_asset_id."""
    mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    filename = os.path.basename(filepath)

    # Step 1: Prepare upload
    resp = requests.post(
        f"{WL_API_BASE}/marble/v1/media-assets:prepare_upload",
        headers=_wl_headers(api_key),
        json={"mime_type": mime, "file_name": filename},
    )
    resp.raise_for_status()
    data = resp.json()
    asset_id = data["media_asset_id"]
    upload_url = data["file_upload_url"]

    # Step 2: Upload file
    with open(filepath, "rb") as f:
        put_resp = requests.put(
            upload_url,
            headers={"Content-Type": mime},
            data=f,
        )
        put_resp.raise_for_status()

    return asset_id


def _poll_operation(api_key: str, operation_name: str, timeout: int = 600) -> dict:
    """Poll an operation until done. Returns the final operation response."""
    op_id = operation_name.split("/")[-1] if "/" in operation_name else operation_name
    start = time.time()
    interval = 3

    while time.time() - start < timeout:
        resp = requests.get(
            f"{WL_API_BASE}/marble/v1/operations/{op_id}",
            headers=_wl_headers(api_key),
        )
        resp.raise_for_status()
        op = resp.json()

        if op.get("done"):
            return op

        status = op.get("metadata", {}).get("progress", {}).get("status", "")
        if status == "SUCCEEDED":
            return op
        if status == "FAILED":
            desc = op.get("metadata", {}).get("progress", {}).get("description", "Unknown error")
            raise RuntimeError(f"World Labs generation failed: {desc}")

        elapsed = time.time() - start
        if elapsed < 30:
            interval = 3
        elif elapsed < 120:
            interval = 5
        else:
            interval = 10

        time.sleep(interval)

    raise TimeoutError(f"World Labs generation timed out after {timeout}s")


def _get_world(api_key: str, world_id: str) -> dict:
    """Fetch world data including splat URLs."""
    resp = requests.get(
        f"{WL_API_BASE}/marble/v1/worlds/{world_id}",
        headers=_wl_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json()


def _download_file(url: str, dest_path: str, api_key: str = None):
    """Download a file from a URL."""
    headers = {}
    if api_key:
        headers["WLT-Api-Key"] = api_key
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def _image_tensor_to_file(image: torch.Tensor, index: int = 0) -> str:
    """Save an IMAGE tensor frame to a temp PNG, return path."""
    frame = (image[index].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    tmp = tempfile.mktemp(suffix=".png", prefix="yak_wl_")
    Image.fromarray(frame).save(tmp)
    return tmp


# ─────────────────────────────────────────────────────────────────────────────
# YAKWorldLabsGenerate — supports text, image, video, multi-view
# ─────────────────────────────────────────────────────────────────────────────

class YAKWorldLabsGenerate:
    """
    Generate a 3D world using the World Labs API.
    Supports text, image, video, and multi-view (up to 4 views) inputs.
    Downloads the resulting .spz splat file and shows it in a 3D viewport.
    """

    CATEGORY = "YAK/WorldLabs"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("spz_path", "world_id", "success", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "World Labs API key (WLT-Api-Key)"
                }),
                "input_mode": (["text", "image", "video", "multi_view"], {
                    "default": "text",
                    "tooltip": "Type of input for generation"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text description of the world to generate"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image (for image mode)"
                }),
                "image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to input image file (alternative to IMAGE input)"
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to input video file (for video mode)"
                }),
                "view_front": ("IMAGE", {"tooltip": "Front view (0 degrees) for multi-view"}),
                "view_right": ("IMAGE", {"tooltip": "Right view (90 degrees) for multi-view"}),
                "view_back": ("IMAGE", {"tooltip": "Back view (180 degrees) for multi-view"}),
                "view_left": ("IMAGE", {"tooltip": "Left view (270 degrees) for multi-view"}),
                "model": (["Marble 0.1-plus", "Marble 0.1-mini"], {
                    "default": "Marble 0.1-plus",
                    "tooltip": "World Labs model to use"
                }),
                "resolution": (["500k", "100k", "full_res"], {
                    "default": "500k",
                    "tooltip": "Splat resolution to download"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output folder for .spz file (default: temp directory)"
                }),
                "display_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Display name for the generated world"
                }),
                "timeout": ("INT", {
                    "default": 600,
                    "min": 60,
                    "max": 1800,
                    "tooltip": "Max seconds to wait for generation"
                }),
            },
        }

    def generate(
        self,
        api_key: str,
        input_mode: str,
        text_prompt: str = "",
        image=None,
        image_path: str = "",
        video_path: str = "",
        view_front=None,
        view_right=None,
        view_back=None,
        view_left=None,
        model: str = "Marble 0.1-plus",
        resolution: str = "500k",
        output_dir: str = "",
        display_name: str = "",
        timeout: int = 600,
    ):
        _require_requests()

        if not api_key or not api_key.strip():
            return self._fail("ERROR: API key is required.")

        api_key = api_key.strip()
        log_lines = []

        try:
            # Build world_prompt based on input_mode
            world_prompt = self._build_prompt(
                api_key, input_mode, text_prompt,
                image, image_path, video_path,
                view_front, view_right, view_back, view_left,
                log_lines,
            )

            # Start generation
            log_lines.append(f"Starting generation (model: {model})...")
            gen_body = {"model": model, "world_prompt": world_prompt}
            if display_name and display_name.strip():
                gen_body["display_name"] = display_name.strip()

            resp = requests.post(
                f"{WL_API_BASE}/marble/v1/worlds:generate",
                headers=_wl_headers(api_key),
                json=gen_body,
            )
            resp.raise_for_status()
            gen_data = resp.json()

            op_name = gen_data.get("name", "")
            world_id = gen_data.get("metadata", {}).get("world_id", "")
            log_lines.append(f"Generation started. World ID: {world_id}")

            # Poll until done
            log_lines.append("Polling for completion...")
            op_result = _poll_operation(api_key, op_name, timeout=timeout)
            log_lines.append("Generation complete!")

            # Get world data
            if not world_id:
                world_id = op_result.get("metadata", {}).get("world_id", "")

            world = _get_world(api_key, world_id)

            # Find .spz URL at requested resolution
            spz_url = self._find_spz_url(world, resolution)
            if not spz_url:
                log_lines.append(f"WARNING: no .spz at resolution '{resolution}', trying any available...")
                spz_url = self._find_spz_url(world, None)

            if not spz_url:
                return self._fail("ERROR: no .spz file available in generation result.", log_lines)

            # Download .spz
            if output_dir and output_dir.strip():
                out_path = output_dir.strip()
            else:
                out_path = tempfile.mkdtemp(prefix="yak_wl_out_")
            os.makedirs(out_path, exist_ok=True)

            spz_filename = f"{world_id}_{resolution}.spz"
            spz_path = os.path.join(out_path, spz_filename)
            log_lines.append(f"Downloading .spz ({resolution})...")
            _download_file(spz_url, spz_path, api_key)
            log_lines.append(f"Saved to: {spz_path}")

            # Build viewport data
            viewport_data = {
                "meshes": [],
                "splats": [spz_path],
                "bg_color": "#1a1a1a",
                "grid": True,
            }

            log = "\n".join(log_lines)
            return {
                "ui": {"viewport_data": [json.dumps(viewport_data)]},
                "result": (spz_path, world_id, True, log),
            }

        except Exception as e:
            log_lines.append(f"ERROR: {e}")
            return self._fail("\n".join(log_lines))

    def _build_prompt(
        self, api_key, input_mode, text_prompt,
        image, image_path, video_path,
        view_front, view_right, view_back, view_left,
        log_lines,
    ):
        """Build the world_prompt dict based on input_mode."""

        if input_mode == "text":
            if not text_prompt or not text_prompt.strip():
                raise ValueError("Text prompt is required for text mode.")
            return {"type": "text", "text_prompt": text_prompt.strip()}

        elif input_mode == "image":
            # Get image file path
            img_file = None
            if image is not None:
                img_file = _image_tensor_to_file(image)
            elif image_path and image_path.strip() and os.path.isfile(image_path.strip()):
                img_file = image_path.strip()

            if not img_file:
                raise ValueError("An image input is required for image mode.")

            log_lines.append(f"Uploading image: {os.path.basename(img_file)}")
            asset_id = _upload_media(api_key, img_file)
            log_lines.append(f"Uploaded. Asset ID: {asset_id}")

            prompt = {
                "type": "image",
                "image_prompt": {
                    "source": {"file_asset": {"media_asset_id": asset_id}},
                },
            }
            if text_prompt and text_prompt.strip():
                prompt["text_prompt"] = text_prompt.strip()
            return prompt

        elif input_mode == "video":
            if not video_path or not video_path.strip() or not os.path.isfile(video_path.strip()):
                raise ValueError("A valid video file path is required for video mode.")

            vid_file = video_path.strip()
            log_lines.append(f"Uploading video: {os.path.basename(vid_file)}")
            asset_id = _upload_media(api_key, vid_file)
            log_lines.append(f"Uploaded. Asset ID: {asset_id}")

            prompt = {
                "type": "video",
                "video_prompt": {
                    "source": {"file_asset": {"media_asset_id": asset_id}},
                },
            }
            if text_prompt and text_prompt.strip():
                prompt["text_prompt"] = text_prompt.strip()
            return prompt

        elif input_mode == "multi_view":
            views = []
            view_inputs = [
                (view_front, 0, "front"),
                (view_right, 90, "right"),
                (view_back, 180, "back"),
                (view_left, 270, "left"),
            ]

            for view_tensor, azimuth, label in view_inputs:
                if view_tensor is not None:
                    img_file = _image_tensor_to_file(view_tensor)
                    log_lines.append(f"Uploading {label} view...")
                    asset_id = _upload_media(api_key, img_file)
                    views.append({
                        "source": {"file_asset": {"media_asset_id": asset_id}},
                        "azimuth_degree": azimuth,
                    })

            if len(views) < 2:
                raise ValueError("Multi-view mode requires at least 2 views.")

            log_lines.append(f"Uploaded {len(views)} views.")

            prompt = {
                "type": "images",
                "images_prompt": {"images": views},
            }
            if text_prompt and text_prompt.strip():
                prompt["text_prompt"] = text_prompt.strip()
            return prompt

        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

    @staticmethod
    def _find_spz_url(world: dict, resolution: str | None) -> str | None:
        """Extract .spz URL from world data at the requested resolution."""
        spz_list = (
            world.get("assets", {})
            .get("splats", {})
            .get("spz_urls", [])
        )
        if not spz_list:
            return None

        if resolution:
            for entry in spz_list:
                if entry.get("resolution") == resolution:
                    return entry.get("url")

        # Fallback: return first available
        return spz_list[0].get("url") if spz_list else None

    @staticmethod
    def _fail(msg: str, log_lines: list | None = None):
        log = "\n".join(log_lines) if log_lines else msg
        viewport_data = {"meshes": [], "splats": [], "bg_color": "#1a1a1a", "grid": True}
        return {
            "ui": {"viewport_data": [json.dumps(viewport_data)]},
            "result": ("", "", False, log),
        }


# ─────────────────────────────────────────────────────────────────────────────
# YAKWorldLabsViewer — view .spz results
# ─────────────────────────────────────────────────────────────────────────────

class YAKWorldLabsViewer:
    """
    View a .spz Gaussian splat file from World Labs in a 3D viewport.
    """

    CATEGORY = "YAK/WorldLabs"
    FUNCTION = "view"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .spz Gaussian splat file"
                }),
            },
            "optional": {
                "bg_color": ("STRING", {
                    "default": "#1a1a1a",
                    "multiline": False,
                    "tooltip": "Viewport background colour (hex)"
                }),
                "grid": ("BOOLEAN", {"default": True}),
            },
        }

    def view(self, spz_path: str, bg_color: str = "#1a1a1a", grid: bool = True):
        splats = []
        if spz_path and spz_path.strip() and os.path.isfile(spz_path.strip()):
            splats.append(os.path.normpath(spz_path.strip()))

        viewport_data = {
            "meshes": [],
            "splats": splats,
            "bg_color": bg_color,
            "grid": grid,
        }

        return {"ui": {"viewport_data": [json.dumps(viewport_data)]}, "result": ()}
