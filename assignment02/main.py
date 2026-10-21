"""Generate reproducible Assignment 2 results; no GPU or third-party packages."""
import argparse
import csv
import json
from pathlib import Path
from graph import load_graph
from arithmetic import arithmetic_intensity, _input_dtype, _layer_parameters
from graph import dtype_bytes


def traffic_breakdown(graph):
    rows=[]; producers={}; previous=None
    for layer in graph:
        dtype=_input_dtype(layer,producers,previous,graph.precision)
        streams=2 if layer.kind=='add' and len(layer.reads)==2 else 1
        rows.append({'layer':layer.name,'input_dtype':dtype,
                     'input_bytes':float(streams*layer.in_elements*dtype_bytes(dtype)),
                     'weight_bytes':float(_layer_parameters(layer)*dtype_bytes(layer.weight_dtype)),
                     'buffer_bytes':float(2*layer.out_shape[0]*4 if layer.kind=='bn' else 0),
                     'output_bytes':float(layer.out_elements*dtype_bytes(layer.act_dtype))})
        producers[layer.name]=layer; previous=layer
    return rows


def run(model_path, output_dir):
    graph=load_graph(model_path)
    result=arithmetic_intensity(graph)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'intensity_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    breakdown=traffic_breakdown(graph)
    with (output_dir/'layer_results.csv').open('w',newline='',encoding='utf-8') as stream:
        keys=['layer','input_dtype','flops','bytes','intensity','input_bytes','weight_bytes','buffer_bytes','output_bytes']
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        for row in breakdown:
            writer.writerow(dict(row,**result['per_layer'][row['layer']]))
    print(f"{graph.name}: {result['total_flops']:,} FLOPs / {result['total_bytes']:,.0f} bytes = {result['value']} FLOPs/byte")
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=root/'model.json')
    parser.add_argument('--output-dir',type=Path,default=root/'results')
    args=parser.parse_args(); run(args.model,args.output_dir)
