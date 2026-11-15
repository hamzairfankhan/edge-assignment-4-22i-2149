"""Export seven trained classifiers and validate CPU ONNX inference on full test sets."""
import json,csv
import torch,numpy as np
import onnx,onnxruntime as ort
from inference import ROOT,NAMES,load_model
from benchmark import raw_test
from q1_common import preprocess
from nas_search import resize
if __name__=='__main__':
    torch.set_num_threads(4);records=[];predictions=[];cached={ds:raw_test(ds) for ds in ['CIFAR-100','CIFAR-10']}
    for name in NAMES:
        model,res,ds=load_model(name);path=ROOT/'models'/(name+'.onnx');x=torch.zeros(1,3,res,res)
        torch.onnx.export(model,x,str(path),input_names=['input'],output_names=['logits'],opset_version=17,dynamo=False,dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}})
        onnx.checker.check_model(onnx.load(str(path)));options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);raw,y=cached[ds];pp=[];max_error=0.
        reference=np.load(ROOT/'results'/(name+'_reference_logits.npy'))
        mean=torch.tensor([.4914,.4822,.4465]).view(1,3,1,1);std=torch.tensor([.247,.2435,.2616]).view(1,3,1,1)
        for i in range(0,len(y),128):
            r=raw[i:i+128];xb=preprocess(r) if ds=='CIFAR-100' else resize((r.float()/255-mean)/std,res)
            z=session.run(None,{'input':xb.numpy()})[0];pp.extend(z.argmax(1).tolist());max_error=max(max_error,float(np.max(np.abs(z-reference[i:i+len(z)]))))
        correct=sum(a==b for a,b in zip(pp,y));row=dict(model=name,dataset=ds,test_count=len(y),correct=correct,accuracy_pct=100*correct/len(y),max_absolute_logit_difference=max_error,onnx_bytes=path.stat().st_size)
        records.append(row);print(json.dumps(row),flush=True)
        predictions.extend(dict(model=name,dataset=ds,test_index=i,true_label=y[i],prediction=pp[i],correct=int(y[i]==pp[i])) for i in range(len(y)))
    (ROOT/'results/onnx_validation.json').write_text(json.dumps(records,indent=2))
    for filename,rows in [('onnx_validation.csv',records),('onnx_predictions.csv',predictions)]:
        with (ROOT/'results'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
