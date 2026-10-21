"""Pre-lab predictions: arithmetic only, no camera measurements."""
import json
from pathlib import Path
from frames import (summarise_formats, bytes_per_second, conversion_bytes,
                    resize_bytes, normalise_bytes)


def predict(width=1920, height=1080, fps=30.):
    return {
        "status": "computed", "source": "Lab 7 packed-format arithmetic",
        "conditions": {"width":width,"height":height,"assumed_fps":fps},
        "format_comparison": summarise_formats(width,height),
        "bandwidth_bytes_per_sec": {
            "bgr_live_stream":bytes_per_second(width,height,'bgr',fps),
            "nv12_live_stream":bytes_per_second(width,height,'nv12',fps)},
        "pipeline_stage_costs": {
            "nv12_to_bgr_conversion_bytes":conversion_bytes(width,height,'nv12','bgr'),
            "resize_to_224x224_rgb_bytes":resize_bytes(width,height,224,224,'rgb'),
            "fp32_normalization_tensor_bytes":normalise_bytes(width,height)},
        "after_resize_normalization_bytes":normalise_bytes(224,224),
        "hardware_status": "not measured by this prediction script",
        "notes": ["Packed payload estimate; excludes row pitch, headers, caching, and additional copies.",
                  "Stage costs assume one full input read and one output write; stages are independent examples.",
                  "FPS is assumed, not measured. No camera image is produced."]}


if __name__ == '__main__':
    path=Path(__file__).with_name('prelab_predictions.json')
    path.write_text(json.dumps(predict(),indent=2)+'\n',encoding='utf-8')
    print(f'Predictions saved to {path}')
