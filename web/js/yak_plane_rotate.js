/**
 * YAK Plane Rotate — ComfyUI extension
 *
 * Embeds an interactive 3D canvas (plane_editor.html) in the YAKPlaneRotate node.
 * The input image is textured onto a plane; the user rotates it with a gizmo. On
 * every rotation the editor renders the frame at the image's native resolution
 * and POSTs it to /yak/plane_save, and we store the returned path in the node's
 * (hidden) `captured` widget so the Python side can read it back on the next run.
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EDITOR_HTML_PATH = new URL("../html/plane_editor.html", import.meta.url).href;
const MIN_WIDTH = 500;
const MIN_HEIGHT = 480;

function attachPlaneEditor(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    onNodeCreated?.apply(this, arguments);

    this.size = [MIN_WIDTH, MIN_HEIGHT];

    // Hide the internal `captured` widget — it only carries the rendered frame path.
    const capWidget = this.widgets?.find((w) => w.name === "captured");
    if (capWidget) {
      capWidget.type = "hidden";
      capWidget.computeSize = () => [0, -4];
      this._yakCapWidget = capWidget;
    }

    // Iframe overlay holding the Three.js editor.
    const container = document.createElement("div");
    container.style.cssText =
      "position:absolute;pointer-events:auto;overflow:hidden;border-radius:4px;border:1px solid #2a2a2a;display:none;z-index:10;";

    const iframe = document.createElement("iframe");
    iframe.src = EDITOR_HTML_PATH;
    iframe.style.cssText = "width:100%;height:100%;border:none;";
    container.appendChild(iframe);
    document.body.appendChild(container);

    this._yakContainer = container;
    this._yakIframe = iframe;
    this._yakPending = null; // texture message queued until the iframe loads

    iframe.addEventListener("load", () => {
      if (this._yakPending) {
        iframe.contentWindow.postMessage(this._yakPending, "*");
      }
    });

    // Receive rendered frames from this node's iframe and persist them.
    this._yakMsgHandler = async (event) => {
      if (event.source !== iframe.contentWindow) return;
      const data = event.data;
      if (!data || data.type !== "yak_plane_capture") return;
      try {
        const res = await api.fetchApi("/yak/plane_save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ b64: data.b64 }),
        });
        const json = await res.json();
        if (json.filepath && this._yakCapWidget) {
          this._yakCapWidget.value = json.filepath;
        }
      } catch (e) {
        console.error("YAK Plane: failed to save capture", e);
      }
    };
    window.addEventListener("message", this._yakMsgHandler);

    // Custom widget that positions the iframe overlay above the node canvas.
    this.addCustomWidget({
      type: "YAK_PLANE_EDITOR",
      name: "plane_editor_widget",
      draw(ctx, node, widgetWidth, widgetY) {
        if (!node._yakContainer) return;
        const t = ctx.getTransform();
        const scale = t.a;
        const x = node.pos[0] * scale + t.e;
        const y = (node.pos[1] + widgetY) * scale + t.f;
        const w = widgetWidth * scale;
        const h = (node.size[1] - widgetY) * scale;

        const rect = app.canvas.canvas.getBoundingClientRect();
        const c = node._yakContainer.style;
        c.left = rect.left + x + "px";
        c.top = rect.top + y + "px";
        c.width = w + "px";
        c.height = Math.max(h, 100) + "px";
        c.display = node.flags?.collapsed ? "none" : "block";
      },
      computeSize() {
        return [MIN_WIDTH, MIN_HEIGHT - 120];
      },
    });
  };

  // When the node runs, Python returns the texture path + native size; hand it
  // to the editor so the plane gets the latest input image.
  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    onExecuted?.apply(this, arguments);
    const raw = message?.plane_data?.[0];
    if (!raw) return;
    const [path, w, h] = raw.split("|");
    const msg = {
      type: "yak_plane_set_texture",
      url: `/yak/viewfile?filepath=${encodeURIComponent(path)}`,
      width: parseInt(w, 10),
      height: parseInt(h, 10),
    };
    this._yakPending = msg;
    if (this._yakIframe?.contentWindow) {
      this._yakIframe.contentWindow.postMessage(msg, "*");
    }
  };

  const onResize = nodeType.prototype.onResize;
  nodeType.prototype.onResize = function (size) {
    onResize?.apply(this, arguments);
    size[0] = Math.max(size[0], MIN_WIDTH);
    size[1] = Math.max(size[1], MIN_HEIGHT);
  };

  const onCollapse = nodeType.prototype.onCollapse;
  nodeType.prototype.onCollapse = function () {
    onCollapse?.apply(this, arguments);
    if (this._yakContainer) this._yakContainer.style.display = "none";
  };

  const onRemoved = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function () {
    onRemoved?.apply(this, arguments);
    if (this._yakMsgHandler) window.removeEventListener("message", this._yakMsgHandler);
    if (this._yakContainer) {
      this._yakContainer.remove();
      this._yakContainer = null;
      this._yakIframe = null;
    }
  };
}

app.registerExtension({
  name: "YAK.PlaneRotate",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "YAKPlaneRotate") {
      attachPlaneEditor(nodeType);
    }
  },
});
