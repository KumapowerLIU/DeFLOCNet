const canvas = element.querySelector("[data-edit-canvas]");
const toolButtons = [...element.querySelectorAll("[data-tool]")];
const eraseButton = element.querySelector("[data-erase]");
const undoButton = element.querySelector("[data-undo]");
const brushSize = element.querySelector("[data-brush-size]");
const brushColor = element.querySelector("[data-brush-color]");
const brushLabel = element.querySelector("[data-brush-label]");
const colorControls = element.querySelector("[data-color-controls]");
const empty = element.querySelector("[data-empty]");
const emptyText = empty.textContent;
const size = 256;
const view = canvas.getContext("2d");
const sizes = {mask: 20, sketch: 3, color: 6};
const history = [];
let tool = "mask";
let erasing = false;
let source = null;
let revision = null;
let loadingRevision = null;
let stroke = null;
let drawPending = false;

function buffer() {
  const layer = document.createElement("canvas");
  layer.width = layer.height = size;
  return layer.getContext("2d", {willReadFrequently: true});
}

const layers = {mask: buffer(), sketch: buffer(), color: buffer()};
const guides = buffer();
canvas.style.touchAction = "none";

function toolbar() {
  const ready = source !== null;
  for (const button of toolButtons) {
    button.disabled = !ready;
    button.setAttribute("aria-pressed", String(button.dataset.tool === tool));
  }
  eraseButton.disabled = !ready;
  eraseButton.setAttribute("aria-pressed", String(erasing));
  undoButton.disabled = !ready || history.length === 0;
  brushSize.disabled = !ready;
  brushSize.value = sizes[tool];
  brushLabel.textContent = `${sizes[tool]} px`;
  colorControls.hidden = tool !== "color";
  brushColor.disabled = !ready || tool !== "color";
  empty.hidden = ready;
  canvas.style.cursor = ready ? "crosshair" : "default";
}

function render() {
  drawPending = false;
  view.clearRect(0, 0, size, size);
  if (source === null) return;
  view.drawImage(source, 0, 0);
  view.drawImage(layers.mask.canvas, 0, 0);
  guides.clearRect(0, 0, size, size);
  guides.drawImage(layers.sketch.canvas, 0, 0);
  guides.drawImage(layers.color.canvas, 0, 0);
  guides.globalCompositeOperation = "destination-in";
  guides.drawImage(layers.mask.canvas, 0, 0);
  guides.globalCompositeOperation = "source-over";
  view.drawImage(guides.canvas, 0, 0);
}

function requestRender() {
  if (!drawPending) {
    drawPending = true;
    requestAnimationFrame(render);
  }
}

function publish() {
  props.value = JSON.stringify({
    revision,
    mask: layers.mask.canvas.toDataURL("image/png"),
    sketch: layers.sketch.canvas.toDataURL("image/png"),
    color: layers.color.canvas.toDataURL("image/png"),
  });
}

function loadPng(url) {
  return new Promise((resolve, reject) => {
    if (typeof url !== "string" || !url.startsWith("data:image/png;base64,")) {
      reject(new Error("Expected a PNG canvas image"));
      return;
    }
    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth === size && image.naturalHeight === size) resolve(image);
      else reject(new Error("Unexpected canvas image dimensions"));
    };
    image.onerror = reject;
    image.src = url;
  });
}

async function loadSource(value) {
  let state;
  try {
    state = typeof value === "string" ? JSON.parse(value) : value;
  } catch {
    return;
  }
  // The only values that reload the image originate from server resets.
  if (!state?.background || !state.revision || state.revision === loadingRevision) return;
  const loading = state.revision;
  loadingRevision = loading;
  source = revision = null;
  stroke = null;
  history.length = 0;
  tool = "mask";
  erasing = false;
  for (const layer of Object.values(layers)) layer.clearRect(0, 0, size, size);
  empty.textContent = "Preparing canvas…";
  toolbar();
  requestRender();
  try {
    const [background, mask] = await Promise.all([
      loadPng(state.background),
      state.mask ? loadPng(state.mask) : null,
    ]);
    if (loadingRevision !== loading) return;
    if (mask !== null) {
      const context = layers.mask;
      context.drawImage(mask, 0, 0);
      context.globalCompositeOperation = "source-in";
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, size, size);
      context.globalCompositeOperation = "source-over";
    }
    source = background;
    revision = loading;
    empty.textContent = emptyText;
  } catch {
    if (loadingRevision !== loading) return;
    empty.textContent = "Could not load the canvas. Choose the image again.";
  }
  toolbar();
  requestRender();
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: (event.clientX - rect.left) * size / rect.width,
          y: (event.clientY - rect.top) * size / rect.height};
}

function paint(from, to) {
  const context = layers[stroke.tool];
  context.globalCompositeOperation = stroke.erase ? "destination-out" : "source-over";
  context.strokeStyle = context.fillStyle = stroke.color;
  context.lineWidth = stroke.size;
  context.lineCap = "round";
  context.beginPath();
  if (from === null) {
    context.arc(to.x, to.y, stroke.size / 2, 0, 2 * Math.PI);
    context.fill();
  } else {
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
  }
  context.globalCompositeOperation = "source-over";
}

function extend(event) {
  const samples = event.getCoalescedEvents?.() ?? [];
  for (const sample of samples.length ? samples : [event]) {
    const next = point(sample);
    if (next.x !== stroke.last.x || next.y !== stroke.last.y) {
      paint(stroke.last, next);
      stroke.last = next;
    }
  }
  requestRender();
}

canvas.addEventListener("pointerdown", (event) => {
  if (source === null || stroke !== null || event.button !== 0) return;
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  stroke = {
    id: event.pointerId,
    tool,
    last: point(event),
    size: sizes[tool],
    erase: erasing,
    color: tool === "mask" ? "#ffffff" : tool === "sketch" ? "#000000" : brushColor.value,
    before: layers[tool].getImageData(0, 0, size, size),
  };
  paint(null, stroke.last);
  requestRender();
});

canvas.addEventListener("pointermove", (event) => {
  if (stroke?.id === event.pointerId) extend(event);
});

canvas.addEventListener("pointerup", (event) => {
  if (stroke?.id !== event.pointerId) return;
  extend(event);
  history.push({tool: stroke.tool, before: stroke.before});
  if (history.length > 16) history.shift();
  stroke = null;
  toolbar();
  publish();
});

function cancelStroke(event) {
  if (stroke?.id !== event.pointerId) return;
  layers[stroke.tool].putImageData(stroke.before, 0, 0);
  stroke = null;
  requestRender();
}

canvas.addEventListener("pointercancel", cancelStroke);
canvas.addEventListener("lostpointercapture", cancelStroke);

for (const button of toolButtons) {
  button.addEventListener("click", () => {
    tool = button.dataset.tool;
    erasing = false;
    toolbar();
  });
}
eraseButton.addEventListener("click", () => {
  erasing = !erasing;
  toolbar();
});
undoButton.addEventListener("click", () => {
  if (stroke !== null || history.length === 0) return;
  const entry = history.pop();
  layers[entry.tool].putImageData(entry.before, 0, 0);
  toolbar();
  requestRender();
  publish();
});
brushSize.addEventListener("input", () => {
  sizes[tool] = Number(brushSize.value);
  brushLabel.textContent = `${sizes[tool]} px`;
});

toolbar();
let currentValue = props.value;
loadSource(currentValue);
if (typeof watch === "function") {
  watch("value", () => loadSource(props.value));
} else {
  // Gradio 6.9 updates the props proxy without providing a watch callback.
  const sync = setInterval(() => {
    if (!element.isConnected) {
      clearInterval(sync);
    } else if (props.value !== currentValue) {
      currentValue = props.value;
      loadSource(currentValue);
    }
  }, 100);
}
