"""
YAK — Background Remove node using BiRefNet (state-of-the-art).
Fast, high-quality background removal — no mask input needed.
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

MODEL_SIZE = 1024


class YAKBackgroundRemove:
    """
    Remove background from images using BiRefNet.
    Works on any image — no green screen or mask needed.
    Model auto-downloads from HuggingFace on first run.
    """

    CATEGORY = "YAK"
    FUNCTION = "remove"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("foreground", "alpha", "mask", "success", "log")
    OUTPUT_NODE = True

    _model = None
    _device = None

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
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for inference"
                }),
            },
        }

    @classmethod
    def _load_model(cls, device: str):
        """Load BiRefNet model, cached across runs."""
        if cls._model is not None and cls._device == device:
            return cls._model

        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        model.to(device)
        model.eval()
        cls._model = model
        cls._device = device
        return model

    def remove(
        self,
        images: torch.Tensor,
        background: str = "green",
        device: str = "auto",
    ):
        h, w = images.shape[1], images.shape[2]
        n_frames = images.shape[0]
        blank = torch.zeros(1, h, w, 3)
        mask_blank = torch.zeros(1, h, w)

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        else:
            dev = device

        log_lines = [f"Device: {dev}"]

        try:
            from torchvision.transforms.functional import normalize
        except ImportError:
            return (blank, blank, mask_blank, False,
                    "ERROR: torchvision not installed.")

        try:
            model = self._load_model(dev)
            log_lines.append("BiRefNet model loaded")
        except Exception as e:
            return (blank, blank, mask_blank, False, f"ERROR loading model: {e}")

        bg_color = BACKGROUND_COLORS.get(background, BACKGROUND_COLORS["green"])
        bg_color = bg_color.reshape(1, 1, 3)

        fgr_list = []
        alpha_list = []

        try:
            for i in range(n_frames):
                frame_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_np)

                # Preprocess: resize to model size, normalize
                input_img = pil_img.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
                input_tensor = torch.from_numpy(
                    np.array(input_img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

                input_tensor = normalize(
                    input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ).to(dev)

                # Inference
                with torch.no_grad():
                    preds = model(input_tensor)[-1].sigmoid().cpu()

                # Post-process: resize mask back to original
                pred = preds[0, 0].numpy()  # (MODEL_SIZE, MODEL_SIZE)
                mask_pil = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
                mask_pil = mask_pil.resize((w, h), Image.BILINEAR)
                alpha = np.array(mask_pil).astype(np.float32) / 255.0  # (H,W)

                # Composite
                frame_f = images[i].cpu().numpy()  # (H,W,3) float 0-1
                a = alpha[:, :, np.newaxis]
                fgr = frame_f * a + bg_color * (1.0 - a)

                fgr_list.append(np.clip(fgr, 0, 1))
                alpha_list.append(np.clip(alpha, 0, 1))

                if (i + 1) % 10 == 0:
                    log_lines.append(f"Processed {i + 1}/{n_frames}")

            log_lines.append(f"Done: {n_frames} frames")

        except Exception as e:
            log_lines.append(f"ERROR: {e}")
            import traceback
            log_lines.append(traceback.format_exc())
            return (blank, blank, mask_blank, False, "\n".join(log_lines))

        fgr_stack = np.stack(fgr_list, axis=0)
        alpha_stack = np.stack(alpha_list, axis=0)

        fgr_out = torch.from_numpy(fgr_stack).float()
        alpha_3ch = np.repeat(np.expand_dims(alpha_stack, -1), 3, axis=-1)
        alpha_out = torch.from_numpy(alpha_3ch).float()
        mask_out = torch.from_numpy(alpha_stack).float()

        return (fgr_out, alpha_out, mask_out, True, "\n".join(log_lines))
