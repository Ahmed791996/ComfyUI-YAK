"""
CorridorKey — ComfyUI custom node package
https://github.com/nikopueringer/CorridorKey
"""

from .nodes import (
    CorridorKeyCheckSetup,
    CorridorKeyProcessClip,
    CorridorKeyBatchProcess,
    CorridorKeyLoadEXRSequence,
)

NODE_CLASS_MAPPINGS = {
    "CorridorKeyCheckSetup":       CorridorKeyCheckSetup,
    "CorridorKeyProcessClip":      CorridorKeyProcessClip,
    "CorridorKeyBatchProcess":     CorridorKeyBatchProcess,
    "CorridorKeyLoadEXRSequence":  CorridorKeyLoadEXRSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKeyCheckSetup":       "CK Check Setup",
    "CorridorKeyProcessClip":      "CK Process Clip",
    "CorridorKeyBatchProcess":     "CK Batch Process",
    "CorridorKeyLoadEXRSequence":  "CK Load EXR Sequence",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
