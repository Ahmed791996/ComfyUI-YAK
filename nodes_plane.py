"""
YAK — Interactive Plane Rotate node.

Shows the input image on a plane inside a 3D canvas viewport. Rotate the plane
on any axis with the in-node gizmo, then queue the prompt: the rendered frame
(exactly what you see) flows out as IMAGE + MASK.

The browser renders the rotated plane and POSTs the frame to /yak/plane_save,
which writes a PNG to the temp directory and returns its path. That path is
stored in the (hidden) `captured` widget, and this node reads it back, composites
it over `bg_color`, and outputs the result.
"""

import os
import tempfile
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    """(H,W,3|4) float 0-1 tensor -> PIL image."""
    arr = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode=mode)


def _hex_to_rgb(value: str, default=(0, 0, 0)):
    h = (value or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return default
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


class YAKPlaneRotate:
    """
    Display the input image on a plane in a canvas viewport, rotate the plane on
    any axis, and export the rendered frame.

    Workflow: queue once so the image loads onto the plane, rotate it with the
    gizmo in the node, then queue again — the rotated render comes out of the
    IMAGE/MASK outputs. The MASK is the plane's coverage (alpha), so the rotated
    card can be composited downstream.
    """

    CATEGORY = "YAK/3D"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "bg_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "Colour composited behind the rotated plane (hex). The plane's coverage is also exposed on the MASK output.",
                }),
                # Written by the viewport JS (hidden in the UI): path to the
                # rendered PNG. Keep it in INPUT_TYPES so its value reaches Python.
                "captured": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "(internal) path to the rendered frame, set by the viewport",
                }),
            },
        }

    def render(self, image, bg_color="#000000", captured=""):
        # Texture the plane with the first frame of the batch.
        src = image[0] if image.dim() == 4 else image
        h, w = int(src.shape[0]), int(src.shape[1])

        # Save the input frame so the browser viewport can load it onto the plane.
        tex_path = tempfile.mktemp(suffix=".png", prefix="yak_plane_tex_")
        _tensor_to_pil(src).save(tex_path)

        # Tell the viewport which texture + native size to use.
        ui = {"plane_data": [f"{tex_path}|{w}|{h}"]}

        cap = (captured or "").strip()

        if not cap or not os.path.isfile(cap):
            # Nothing rendered yet — pass the input straight through so the graph
            # still runs and the user can see/rotate the plane first.
            rgb = src[..., :3] if src.shape[-1] >= 3 else src.unsqueeze(-1).repeat(1, 1, 3)
            out_img = rgb.unsqueeze(0).float()
            out_mask = torch.ones((1, h, w), dtype=torch.float32)
            return {"ui": ui, "result": (out_img, out_mask)}

        # Decode the rotated render (RGBA PNG with transparent background).
        with Image.open(cap) as im:
            rgba = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0  # (H,W,4)

        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3]

        bg = np.array(_hex_to_rgb(bg_color), dtype=np.float32) / 255.0
        comp = rgb * alpha + bg[None, None, :] * (1.0 - alpha)

        out_img = torch.from_numpy(np.ascontiguousarray(comp)).unsqueeze(0).float()
        out_mask = torch.from_numpy(np.ascontiguousarray(alpha[..., 0])).unsqueeze(0).float()
        return {"ui": ui, "result": (out_img, out_mask)}
