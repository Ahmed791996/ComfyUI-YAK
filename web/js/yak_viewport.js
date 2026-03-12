/**
 * YAK 3D Viewport — ComfyUI extension
 * Creates an iframe-based 3D viewport widget for all YAK nodes that output viewport_data.
 */
import { app } from "../../../scripts/app.js";

const VIEWPORT_HTML_PATH = new URL("../html/viewport.html", import.meta.url).href;
const MIN_WIDTH = 500;
const MIN_HEIGHT = 450;

// All node types that get an embedded 3D viewport
const VIEWPORT_NODES = new Set([
  "YAK3DViewport",
  "YAKSharpGenerate",
  "YAKSharpViewer",
  "YAKWorldLabsGenerate",
  "YAKWorldLabsViewer",
]);

function attachViewport(nodeType, nodeData) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    onNodeCreated?.apply(this, arguments);

    this.size = [MIN_WIDTH, MIN_HEIGHT];

    // Create iframe container
    const container = document.createElement("div");
    container.style.cssText =
      "position:absolute;pointer-events:auto;overflow:hidden;border-radius:4px;border:1px solid #2a2a2a;display:none;z-index:10;";

    const iframe = document.createElement("iframe");
    iframe.src = VIEWPORT_HTML_PATH;
    iframe.style.cssText = "width:100%;height:100%;border:none;";
    iframe.allow = "autoplay";
    container.appendChild(iframe);
    document.body.appendChild(container);

    this._yakContainer = container;
    this._yakIframe = iframe;
    this._yakData = null;

    iframe.addEventListener("load", () => {
      if (this._yakData) {
        iframe.contentWindow.postMessage(this._yakData, "*");
      }
    });

    // Custom widget that positions the iframe overlay
    const widget = {
      type: "YAK_VIEWPORT",
      name: "viewport_widget",
      draw(ctx, node, widgetWidth, widgetY) {
        if (!node._yakContainer) return;

        const transform = ctx.getTransform();
        const scale = transform.a;
        const offsetX = transform.e;
        const offsetY = transform.f;

        const x = node.pos[0] * scale + offsetX;
        const y = (node.pos[1] + widgetY) * scale + offsetY;
        const w = widgetWidth * scale;
        const h = (node.size[1] - widgetY) * scale;

        const canvasEl = app.canvas.canvas;
        const rect = canvasEl.getBoundingClientRect();

        container.style.left = (rect.left + x) + "px";
        container.style.top = (rect.top + y) + "px";
        container.style.width = w + "px";
        container.style.height = Math.max(h, 100) + "px";
        container.style.display = "block";
      },
      computeSize() {
        return [MIN_WIDTH, MIN_HEIGHT - 80];
      },
    };

    this.addCustomWidget(widget);
  };

  // Handle execution result — all viewport nodes return viewport_data via ui
  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    onExecuted?.apply(this, arguments);

    if (message?.viewport_data?.[0]) {
      try {
        const parsed = JSON.parse(message.viewport_data[0]);
        this._yakData = { type: "yak_viewport_data", ...parsed };

        if (this._yakIframe?.contentWindow) {
          this._yakIframe.contentWindow.postMessage(this._yakData, "*");
        }
      } catch (e) {
        console.error("YAK Viewport: failed to parse viewport data", e);
      }
    }
  };

  // Enforce minimum size
  const onResize = nodeType.prototype.onResize;
  nodeType.prototype.onResize = function (size) {
    onResize?.apply(this, arguments);
    size[0] = Math.max(size[0], MIN_WIDTH);
    size[1] = Math.max(size[1], MIN_HEIGHT);
  };

  // Hide when collapsed
  const onCollapse = nodeType.prototype.onCollapse;
  nodeType.prototype.onCollapse = function () {
    onCollapse?.apply(this, arguments);
    if (this._yakContainer) this._yakContainer.style.display = "none";
  };

  // Cleanup on remove
  const onRemoved = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function () {
    onRemoved?.apply(this, arguments);
    if (this._yakContainer) {
      this._yakContainer.remove();
      this._yakContainer = null;
      this._yakIframe = null;
    }
  };
}

app.registerExtension({
  name: "YAK.3DViewport",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (VIEWPORT_NODES.has(nodeData.name)) {
      attachViewport(nodeType, nodeData);
    }
  },
});
