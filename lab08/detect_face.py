"""Lab 8: ONNX face detection and SciPy/Pillow Canny, without OpenCV."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = Path(__file__).with_name("version-RFB-320.onnx")


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """Keep highest-score boxes, suppressing only IoU strictly above threshold."""
    values = np.asarray(box_scores)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("box_scores must have shape (N, 5): x1,y1,x2,y2,score")
    if not 0 <= iou_threshold <= 1 or candidate_size <= 0:
        raise ValueError("IoU threshold must be in [0,1] and candidate_size positive")
    if len(values) == 0:
        return values.copy()
    if not np.isfinite(values).all():
        raise ValueError("box coordinates and scores must be finite")

    # 1. Limit work to the strongest candidates; stable sorting resolves ties.
    order = np.argsort(values[:, 4], kind="stable")[-candidate_size:]
    boxes = values[:, :4]
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    picked = []


    # 2. Pick the strongest box, then discard overlapping alternatives.
    while order.size:
        active = order[-1]
        picked.append(active)
        if (top_k > 0 and len(picked) >= top_k) or order.size == 1:
            break
        remaining = order[:-1]
        lower = np.maximum(boxes[remaining, :2], boxes[active, :2])
        upper = np.minimum(boxes[remaining, 2:], boxes[active, 2:])
        dimensions = np.maximum(0, upper - lower)
        intersection = dimensions[:, 0] * dimensions[:, 1]
        union = areas[remaining] + areas[active] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return values[picked].copy()


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    """Detect faces, draw scored rectangles, and return reproducible metadata."""
    import onnxruntime as ort
    if not 0 <= conf_threshold <= 1:
        raise ValueError("confidence threshold must be in [0,1]")

    # 1. Keep the original image for annotation, resize only the network input.
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    resized = image.resize((320, 240), getattr(Image, "Resampling", Image).BILINEAR)
    pixels = (np.asarray(resized, dtype=np.float32) - 127.0) / 128.0
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...])


    # 2. Request installed providers only. Record actual session providers.
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=providers)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    scores = next(value[0] for value in outputs if value.ndim == 3 and value.shape[-1] == 2)
    boxes = next(value[0] for value in outputs if value.ndim == 3 and value.shape[-1] == 4)


    # 3. Filter face confidence, then merge duplicate predictions with NMS.
    selected = scores[:, 1] >= conf_threshold
    candidates = np.column_stack((boxes[selected], scores[selected, 1]))
    detections = hard_nms(candidates, iou_threshold=0.3)
    draw = ImageDraw.Draw(image)
    findings = []
    for row in detections:
        x1, y1, x2, y2 = row[:4] * np.array([width, height, width, height])
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        coordinates = [int(x1), int(y1), int(x2), int(y2)]
        draw.rectangle(coordinates, outline="red", width=3)
        draw.text((coordinates[0], max(0, coordinates[1] - 12)), f"face {row[4]:.3f}", fill="red")
        findings.append({"box_xyxy": coordinates, "confidence": float(row[4])})
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return {"image_size": [width, height], "input_tensor_shape": list(tensor.shape),
            "providers": session.get_providers(), "confidence_threshold": conf_threshold,
            "iou_threshold": 0.3, "candidate_size": 200, "faces": findings,
            "output": destination.name}


def non_maximum_suppression(magnitude, angle):
    """Thin ridges along the four gradient directions specified by the handout."""
    magnitude = np.asarray(magnitude)
    angle = np.asarray(angle)
    if magnitude.ndim != 2 or magnitude.shape != angle.shape:
        raise ValueError("magnitude and angle must be matching 2D arrays")
    result = np.zeros(magnitude.shape, dtype=np.float32)
    degrees = np.rad2deg(angle) % 180.0

    # Border pixels remain zero because they have no complete neighborhood.
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            direction = degrees[i, j]
            if direction < 22.5 or direction >= 157.5:
                a, b = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                a, b = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                a, b = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                a, b = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] >= a and magnitude[i, j] >= b:
                result[i, j] = magnitude[i, j]
    return result


def hysteresis_thresholding(img, low_thresh, high_thresh):
    """Apply the handout's single-pass, 8-neighbor weak-edge promotion."""
    values = np.asarray(img)
    if values.ndim != 2 or not 0 <= low_thresh <= high_thresh:
        raise ValueError("expected a 2D map and 0 <= low <= high")
    result = np.zeros(values.shape, dtype=np.uint8)
    weak = (values >= low_thresh) & (values < high_thresh)
    result[weak] = 50
    result[values >= high_thresh] = 255

    # Inspect weak pixels once in row-major order, as in the supplied procedure.
    # Earlier promotions may help later pixels; this is not full graph tracking.
    height, width = values.shape
    for i, j in np.argwhere(weak):
        neighbors = result[max(0, i - 1):min(height, i + 2),
                           max(0, j - 1):min(width, j + 2)]
        result[i, j] = 255 if np.any(neighbors == 255) else 0
    return result


def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    """Gaussian -> Sobel -> magnitude/direction -> thinning -> hysteresis."""
    with Image.open(image_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32)
    smooth = gaussian_filter(gray, sigma=1.4)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx, gy = convolve(smooth, kx), convolve(smooth, ky)
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max())
    if maximum:
        magnitude = magnitude / maximum * 255.0
    angle = np.arctan2(gy, gx)
    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(edges).save(destination)
    return edges


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    detection = detect_faces(root / "face.jpg", root / "output_detected.png")
    edges = canny_scipy(root / "face.jpg", root / "edges_scipy.png")
    report = {"face_detection": detection,
              "edge_detection": {"low": 20, "high": 60, "sigma": 1.4,
                                 "edge_pixels": int(np.count_nonzero(edges)),
                                 "total_pixels": int(edges.size),
                                 "output": "edges_scipy.png"}}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
