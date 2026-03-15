"""
YAK — ComfyUI custom nodes: CorridorKey keying, MatAnyone matting, ml-sharp 3D, World Labs generation, 3D viewport
https://github.com/Ahmed791996/ComfyUI-YAK
"""

from .nodes import (
    YAKCheckSetup,
    MaskKey_Green,
    YAKBatchProcess,
    YAKLoadEXRSequence,
    YAK3DViewport,
)
from .nodes_sharp import (
    YAKSharpGenerate,
    YAKSharpViewer,
)
from .nodes_worldlabs import (
    YAKWorldLabsGenerate,
    YAKWorldLabsViewer,
)
from .nodes_matanyone import (
    YAKMatAnyoneGenerate,
    YAKMatAnyoneCheckSetup,
)
from .nodes_rmbg import (
    YAKBackgroundRemove,
)

# Register /yak/viewfile route on import
from . import server_routes  # noqa: F401

NODE_CLASS_MAPPINGS = {
    # CorridorKey
    "YAKCheckSetup":         YAKCheckSetup,
    "MaskKey_Green":         MaskKey_Green,
    "YAKBatchProcess":       YAKBatchProcess,
    "YAKLoadEXRSequence":    YAKLoadEXRSequence,
    # 3D Viewport
    "YAK3DViewport":         YAK3DViewport,
    # ml-sharp
    "YAKSharpGenerate":      YAKSharpGenerate,
    "YAKSharpViewer":        YAKSharpViewer,
    # World Labs
    "YAKWorldLabsGenerate":  YAKWorldLabsGenerate,
    "YAKWorldLabsViewer":    YAKWorldLabsViewer,
    # MatAnyone
    "YAKMatAnyoneGenerate":     YAKMatAnyoneGenerate,
    "YAKMatAnyoneCheckSetup":   YAKMatAnyoneCheckSetup,
    # Background Remove
    "YAKBackgroundRemove":      YAKBackgroundRemove,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # CorridorKey
    "YAKCheckSetup":         "YAK Check Setup",
    "MaskKey_Green":         "YAK MaskKey Green",
    "YAKBatchProcess":       "YAK Batch Process",
    "YAKLoadEXRSequence":    "YAK Load EXR Sequence",
    # 3D Viewport
    "YAK3DViewport":         "YAK 3D Viewport",
    # ml-sharp
    "YAKSharpGenerate":      "YAK Sharp Generate",
    "YAKSharpViewer":        "YAK Sharp Viewer",
    # World Labs
    "YAKWorldLabsGenerate":  "YAK World Labs Generate",
    "YAKWorldLabsViewer":    "YAK World Labs Viewer",
    # MatAnyone
    "YAKMatAnyoneGenerate":     "YAK MatAnyone Generate",
    "YAKMatAnyoneCheckSetup":   "YAK MatAnyone Check Setup",
    # Background Remove
    "YAKBackgroundRemove":      "YAK Background Remove",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
