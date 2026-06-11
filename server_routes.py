"""
YAK — Custom aiohttp routes for serving 3D assets to the browser viewport.
"""

import os
import base64
import tempfile
import mimetypes
from aiohttp import web
import server

ALLOWED_EXTENSIONS = {
    ".glb", ".gltf", ".obj", ".fbx", ".ply", ".splat", ".spz",
    ".mtl", ".bin", ".png", ".jpg", ".jpeg", ".hdr",
}

# Register additional MIME types for 3D formats
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/octet-stream", ".splat")
mimetypes.add_type("application/octet-stream", ".spz")
mimetypes.add_type("application/octet-stream", ".ply")


@server.PromptServer.instance.routes.get("/yak/viewfile")
async def yak_view_file(request):
    """Serve a 3D asset file from disk to the browser viewport."""
    filepath = request.rel_url.query.get("filepath", "")

    if not filepath:
        return web.Response(status=400, text="Missing filepath parameter")

    filepath = os.path.normpath(filepath)

    if not os.path.isfile(filepath):
        return web.Response(status=404, text="File not found")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return web.Response(status=403, text=f"File type {ext} not allowed")

    content_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"

    return web.FileResponse(
        filepath,
        headers={"Content-Type": content_type, "Access-Control-Allow-Origin": "*"},
    )


@server.PromptServer.instance.routes.post("/yak/plane_save")
async def yak_plane_save(request):
    """Save a base64 PNG rendered by the YAK Plane Rotate viewport to a temp file.

    Body: {"b64": "<data URL or raw base64>"}
    Returns: {"filepath": "<absolute path>"}
    """
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    b64 = data.get("b64", "")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    if not b64:
        return web.Response(status=400, text="Missing b64 parameter")

    try:
        raw = base64.b64decode(b64)
    except Exception:
        return web.Response(status=400, text="Invalid base64 data")

    filepath = tempfile.mktemp(suffix=".png", prefix="yak_plane_cap_")
    with open(filepath, "wb") as f:
        f.write(raw)

    return web.json_response({"filepath": filepath})
