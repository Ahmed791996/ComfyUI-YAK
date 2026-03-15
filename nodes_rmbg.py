"""
YAK — Background Remove node using briaai/RMBG-1.4.
Fast, single-image background removal — no mask input needed.
"""

import numpy as np
import torch
from PIL import Image

BACKGROUND_COLORS = {
    "green":   np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "blue":    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "red":     np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "white":   np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "black":   np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


class YAKBackgroundRemove:
    """
    Remove background from images using RMBG-1.4.
    Works on any image — no green screen or mask needed.
    Model auto-downloads from HuggingFace on first run.
    """

    CATEGORY = "YAK"
    FUNCTION = "remove"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("foreground", "alpha", "mask", "success", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image(s) — any content, any background"
                }),
            },
            "optional": {
                "background": (list(BACKGROUND_COLORS.keys()), {
                    "default": "green",
                    "tooltip": "Background color for the foreground composite"
                }),
            },
        }

    def remove(
        self,
        images: torch.Tensor,
        background: str = "green",
    ):
        h, w = images.shape[1], images.shape[2]
        n_frames = images.shape[0]
        blank = torch.zeros(1, h, w, 3)
        mask_blank = torch.zeros(1, h, w)

        try:
            from transformers import pipeline
        except ImportError:
            return (blank, blank, mask_blank, False,
                    "ERROR: transformers not installed.\n"
                    "Install with: pip install transformers")

        log_lines = []

        try:
            segmenter = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
            )
            log_lines.append("RMBG-1.4 model loaded")
        except Exception as e:
            return (blank, blank, mask_blank, False, f"ERROR loading model: {e}")

        bg_color = BACKGROUND_COLORS.get(background, BACKGROUND_COLORS["green"])
        bg_color = bg_color.reshape(1, 1, 3)

        fgr_list = []
        alpha_list = []

        for i in range(n_frames):
            frame_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_np)

            try:
                result = segmenter(pil_img)

                # Combine all masks
                combined = np.zeros((h, w), dtype=np.float32)
                for item in result:
                    mask_pil = item["mask"]
                    if mask_pil.size != (w, h):
                        mask_pil = mask_pil.resize((w, h), Image.BILINEAR)
                    mask_arr = np.array(mask_pil).astype(np.float32)
                    if mask_arr.max() > 1:
                        mask_arr = mask_arr / 255.0
                    combined = np.maximum(combined, mask_arr)

                # Composite
                frame_f = images[i].cpu().numpy()  # (H,W,3) float 0-1
                a = combined[:, :, np.newaxis]      # (H,W,1)
                fgr = frame_f * a + bg_color * (1.0 - a)

                fgr_list.append(np.clip(fgr, 0, 1))
                alpha_list.append(np.clip(combined, 0, 1))

            except Exception as e:
                log_lines.append(f"ERROR on frame {i}: {e}")
                fgr_list.append(np.zeros((h, w, 3), dtype=np.float32))
                alpha_list.append(np.zeros((h, w), dtype=np.float32))

            if (i + 1) % 10 == 0:
                log_lines.append(f"Processed {i + 1}/{n_frames}")

        log_lines.append(f"Done: {n_frames} frames")

        fgr_stack = np.stack(fgr_list, axis=0)
        alpha_stack = np.stack(alpha_list, axis=0)

        fgr_out = torch.from_numpy(fgr_stack).float()
        alpha_3ch = np.repeat(np.expand_dims(alpha_stack, -1), 3, axis=-1)
        alpha_out = torch.from_numpy(alpha_3ch).float()
        mask_out = torch.from_numpy(alpha_stack).float()

        return (fgr_out, alpha_out, mask_out, True, "\n".join(log_lines))
