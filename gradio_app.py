import os
import time
import numpy as np
import gradio as gr
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf

# ------------------ Load model ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_cnn_model_fixed_keras215.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ------------------ Helpers ------------------
def to_pil(x):
    if x is None:
        return None

    # Gradio 6 sketchpad returns dict
    if isinstance(x, dict):
        for k in ("composite", "image", "background"):
            if k in x and x[k] is not None:
                x = x[k]
                break
        if isinstance(x, dict) and "layers" in x and x["layers"]:
            x = x["layers"][-1]

    if isinstance(x, np.ndarray):
        return Image.fromarray(x.astype("uint8"))
    if isinstance(x, Image.Image):
        return x

    try:
        return Image.fromarray(np.array(x).astype("uint8"))
    except Exception:
        return None


def mnist_preprocess(pil: Image.Image, thr=30, blur_radius=0.8):
    img = pil.convert("L")

    # invert if background is light
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    if blur_radius and float(blur_radius) > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    a = np.array(img)
    mask = (a > int(thr)).astype(np.uint8)

    if mask.sum() < 30:
        return None, None, "❌ No digit detected. Draw bigger/thicker."

    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    pad = 10
    h, w = mask.shape
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)

    crop = img.crop((x0, y0, x1 + 1, y1 + 1))

    # resize into 20x20 box
    cw, ch = crop.size
    if ch > cw:
        new_h = 20
        new_w = max(1, int(cw * (20 / ch)))
    else:
        new_w = 20
        new_h = max(1, int(ch * (20 / cw)))

    crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop_resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    x = (np.array(canvas).astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return x, canvas, ""


def pretty_prediction_html(digit=None, conf=None, warn=""):
    if digit is None:
        return """
        <div class="pred-box">
          <div class="pred-title">Predicted Digit</div>
          <div class="pred-digit">?</div>
          <div class="pred-sub">Draw a digit and click <b>Recognize</b></div>
        </div>
        """
    warn_html = f'<div class="pred-warn">{warn}</div>' if warn else ""
    return f"""
    <div class="pred-box">
      <div class="pred-title">Predicted Digit</div>
      <div class="pred-digit">{digit}</div>
      <div class="pred-sub">Confidence: <span class="conf-value">{conf*100:.2f}%</span></div>
      {warn_html}
    </div>
    """


def stats_html(total, best, ms):
    return f"""
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-label">Predictions</div><div class="stat-value">{int(total)}</div></div>
      <div class="stat-card"><div class="stat-label">Best Confidence</div><div class="stat-value">{best*100:.1f}%</div></div>
      <div class="stat-card"><div class="stat-label">Time</div><div class="stat-value">{ms}</div></div>
    </div>
    """

# ------------------ Predict ------------------
def predict_any(inp, thr, blur, total_preds, best_conf):
    pil = to_pil(inp)
    if pil is None:
        return pretty_prediction_html(None, None), None, {}, total_preds, best_conf, stats_html(total_preds, best_conf, "0ms")

    t0 = time.time()
    x, img28, warn = mnist_preprocess(pil, thr=thr, blur_radius=blur)
    if x is None:
        return f"<div class='pred-box'><div class='pred-error'>❌ No digit detected</div></div>", None, {}, total_preds, best_conf, stats_html(total_preds, best_conf, "0ms")

    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    conf = float(np.max(probs))

    elapsed_ms = int((time.time() - t0) * 1000)
    total_preds = int(total_preds) + 1
    best_conf = max(float(best_conf), conf)

    prob_dict = {str(i): float(probs[i]) for i in range(10)}
    return pretty_prediction_html(digit, conf, warn), img28, prob_dict, total_preds, best_conf, stats_html(total_preds, best_conf, f"{elapsed_ms}ms")


# ✅ Blank sketch dict that REALLY clears the sketchpad (Gradio 6 fix)
def blank_sketch(width=650, height=360):
    blank = np.full((height, width, 3), 255, dtype=np.uint8)  # white
    return {"background": blank, "layers": [], "composite": blank}


# ✅ Clear ALL (Sketch + Upload + Outputs + Stats)
def clear_all():
    return (
        blank_sketch(650, 360),          # ✅ sketch clears
        None,                             # upload clears
        pretty_prediction_html(None, None),
        None,                             # processed clears
        {},                               # probs clears
        0,                                # total_preds resets
        0.0,                              # best_conf resets
        stats_html(0, 0.0, "0ms"),        # stats resets
    )

# ------------------ UI + CSS ------------------
CSS = """
body { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important; }
.gradio-container { max-width: 1400px !important; }

.hero {
  padding: 22px 26px; border-radius: 20px; color: white;
  background: linear-gradient(135deg,#f093fb 0%,#f5576c 100%);
  box-shadow: 0 12px 35px rgba(245,87,108,0.35);
  margin-bottom: 16px;
}
.card {
  background: rgba(255,255,255,0.96);
  border-radius: 18px; padding: 14px;
  box-shadow: 0 12px 35px rgba(0,0,0,0.18);
}
.card-title {
  font-size: 18px; font-weight: 900; color: #ffffff !important;
  padding: 10px 14px; border-radius: 14px; margin-bottom: 12px;
  background: linear-gradient(135deg,#3b82f6 0%, #8b5cf6 100%);
}
.pred-box{ background: linear-gradient(135deg,#a8edea 0%,#fed6e3 100%); border-radius: 16px; padding: 18px; text-align: center; }
.pred-title { font-size: 16px; font-weight: 900; color:#1f2a7a; }
.pred-digit { font-size: 84px; font-weight: 1000; color:#4f46e5; }
.pred-sub { font-size: 16px; font-weight: 900; color:#111827; }
.conf-value { color:#111827; font-weight: 1000; font-size: 18px; padding: 2px 8px; border-radius: 10px; background: rgba(255,255,255,0.8); }
.pred-error { font-size: 20px; font-weight: 900; color:#b10f2e; }

.stats-grid{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.stat-card{ background: linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%); border-radius: 14px; padding: 12px; text-align:center; }
.stat-label{ font-size: 12px; font-weight: 900; color:#374151; }
.stat-value{ font-size: 28px; font-weight: 1000; color:#111827; margin-top: 6px; }
"""

with gr.Blocks() as demo:
    total_preds = gr.State(0)
    best_conf = gr.State(0.0)

    gr.HTML("""
    <div class="hero">
      <h1 style="margin:0;font-size:42px;">✨ Handwritten Digit Recognition</h1>
      <p style="margin:8px 0 0 0;">Draw <b>ONE</b> digit (0–9) or upload a single digit image, then click <b>Recognize</b>.</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">✏️ Draw</div>')
                sketch = gr.Sketchpad(label="Draw here", height=360, width=650, value=blank_sketch(650, 360))

                with gr.Row():
                    btn_draw = gr.Button("🔍 Recognize (Draw)", variant="primary")
                    btn_clear_all = gr.Button("🧹 Clear All", variant="secondary")

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">📁 Upload</div>')
                upload = gr.Image(type="pil", label="Upload (single digit)")
                btn_upload = gr.Button("🔍 Recognize (Upload)", variant="primary")

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">⚙️ Settings</div>')
                thr = gr.Slider(0, 120, value=30, step=1, label="Threshold")
                blur = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Blur")

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">🎯 Results</div>')
                out_html = gr.HTML(pretty_prediction_html(None, None))
                out_28 = gr.Image(label="🖼 Processed 28×28 (Model Input)")
                out_probs = gr.Label(num_top_classes=10, label="📊 Confidence Levels")

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">📌 Stats</div>')
                stats = gr.HTML(stats_html(0, 0.0, "0ms"))

    # actions
    btn_draw.click(
        fn=predict_any,
        inputs=[sketch, thr, blur, total_preds, best_conf],
        outputs=[out_html, out_28, out_probs, total_preds, best_conf, stats],
    )

    btn_upload.click(
        fn=predict_any,
        inputs=[upload, thr, blur, total_preds, best_conf],
        outputs=[out_html, out_28, out_probs, total_preds, best_conf, stats],
    )

    # ✅ This clears EVERYTHING including the sketchpad
    btn_clear_all.click(
        fn=clear_all,
        inputs=[],
        outputs=[sketch, upload, out_html, out_28, out_probs, total_preds, best_conf, stats],
    )

demo.launch(css=CSS)
