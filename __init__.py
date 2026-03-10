"""
YAK — ComfyUI custom nodes for CorridorKey green screen keying
https://github.com/Ahmed791996/YAK-nodes
"""

from .nodes import (
    YAKCheckSetup,
    MaskKey_Green,
    YAKBatchProcess,
    YAKLoadEXRSequence,
)

NODE_CLASS_MAPPINGS = {
    "YAKCheckSetup":       YAKCheckSetup,
    "MaskKey_Green":       MaskKey_Green,
    "YAKBatchProcess":     YAKBatchProcess,
    "YAKLoadEXRSequence":  YAKLoadEXRSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YAKCheckSetup":       "YAK Check Setup",
    "MaskKey_Green":       "MaskKey Green",
    "YAKBatchProcess":     "YAK Batch Process",
    "YAKLoadEXRSequence":  "YAK Load EXR Sequence",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
