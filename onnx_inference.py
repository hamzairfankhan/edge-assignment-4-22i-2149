import argparse,json
import torch
import onnxruntime as ort
from inference import ROOT,NAMES,image_tensor
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=NAMES,default='nas_best');p.add_argument('--image',required=True);a=p.parse_args()
    ds='CIFAR-10' if a.model=='nas_best' else 'CIFAR-100'
    res=json.loads((ROOT/'results/best_architecture.json').read_text())['config']['resolution'] if a.model=='nas_best' else 64
    x=image_tensor(a.image,res,ds).numpy();options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/'models'/(a.model+'.onnx')),sess_options=options,providers=['CPUExecutionProvider']);z=session.run(None,{'input':x})[0]
    print(dict(model=a.model,dataset=ds,class_index=int(z.argmax(1)[0]),logits=z.tolist()))
