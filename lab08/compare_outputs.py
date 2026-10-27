"""Compare outputs with instructor images; these are examples, not labeled ground truth."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent

def compare():
    reference=np.asarray(Image.open(root/'sample_edges_scipy.png').convert('L'))>0
    actual=np.asarray(Image.open(root/'edges_scipy.png').convert('L'))>0
    if reference.shape!=actual.shape:
        raise ValueError('edge image dimensions differ')
    intersection=int(np.count_nonzero(reference & actual))
    union=int(np.count_nonzero(reference | actual))
    ref_count=int(np.count_nonzero(reference)); actual_count=int(np.count_nonzero(actual))
    result={'edge_comparison':{'reference_edge_pixels':ref_count,
        'actual_edge_pixels':actual_count,'exact_pixel_match':bool(np.array_equal(reference,actual)),
        'intersection_pixels':intersection,'union_pixels':union,
        'edge_iou':intersection/union if union else 1.,
        'edge_f1':2*intersection/(ref_count+actual_count) if ref_count+actual_count else 1.,
        'pixel_agreement_fraction':float(np.mean(reference==actual))},
        'note':'Reference-image agreement is not accuracy on labeled data; threshold and implementation details can differ.'}
    # Recover the sample's axis-aligned green outline, excluding its small label.
    rgb=np.asarray(Image.open(root/'sample_output_detected.png').convert('RGB'))
    green=(rgb[:,:,1]>200)&(rgb[:,:,0]<40)&(rgb[:,:,2]<40)
    ys=np.flatnonzero(green.sum(axis=1)>100); xs=np.flatnonzero(green.sum(axis=0)>100)
    if len(xs) and len(ys):
        ref_box=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
        boxes=json.loads((root/'results.json').read_text())['face_detection']['faces']
        comparisons=[]
        for record in boxes:
            box=record['box_xyxy']
            w=max(0,min(ref_box[2],box[2])-max(ref_box[0],box[0]))
            h=max(0,min(ref_box[3],box[3])-max(ref_box[1],box[1]))
            intersection=w*h
            union=(ref_box[2]-ref_box[0])*(ref_box[3]-ref_box[1])+(box[2]-box[0])*(box[3]-box[1])-intersection
            comparisons.append({'actual_box':box,'iou_against_sample_outline':intersection/union if union else 0.})
        result['face_comparison']={'sample_outline_xyxy':ref_box,'actual_face_count':len(boxes),'boxes':comparisons,
            'method':'Sample outline inferred from long green borders; not supplied ground-truth boxes.'}
    (root/'sample_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': compare()
