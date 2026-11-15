"""Benchmark six student models and the NAS model on the same Windows CPU."""
from pathlib import Path
import csv,json,os,subprocess,sys,platform
import torch,numpy as np
from torchvision.datasets import CIFAR100,CIFAR10
from inference import ROOT,NAMES,load_model
from q1_common import preprocess
from nas_search import resize
@torch.inference_mode()
def test(model,raw,labels,dataset,resolution):
    pp=[];reference_logits=[]
    mean=torch.tensor([.4914,.4822,.4465]).view(1,3,1,1);std=torch.tensor([.247,.2435,.2616]).view(1,3,1,1)
    for i in range(0,len(labels),128):
        r=raw[i:i+128]
        x=preprocess(r) if dataset=='CIFAR-100' else resize((r.float()/255-mean)/std,resolution)
        z=model(x);pp.extend(z.argmax(1).tolist());reference_logits.append(z.numpy())
    return pp,np.concatenate(reference_logits)
def raw_test(dataset):
    folder=os.environ.get('CIFAR100_ROOT',str(ROOT/'data')) if dataset=='CIFAR-100' else os.environ.get('CIFAR10_ROOT',str(ROOT/'data'))
    d=(CIFAR100 if dataset=='CIFAR-100' else CIFAR10)(folder,train=False,download=True)
    return torch.from_numpy(d.data.transpose(0,3,1,2).copy()),d.targets
if __name__=='__main__':
    torch.set_num_threads(4);rows=[];predictions=[];latencies=[];(ROOT/'results').mkdir(exist_ok=True)
    cached={dataset:raw_test(dataset) for dataset in ['CIFAR-100','CIFAR-10']}
    for name in NAMES:
        model,res,ds=load_model(name);sample=torch.zeros(1,3,res,res);path=ROOT/'models'/(name+'.pt')
        if name!='nas_best': torch.jit.trace(model,sample).save(str(path))
        profile=json.loads(subprocess.check_output([sys.executable,str(ROOT/'cpu_probe.py'),str(path),'--resolution',str(res)],text=True))
        executable=torch.jit.load(str(path)).eval();raw,y=cached[ds];p,logits=test(executable,raw,y,ds,res)
        np.save(ROOT/'results'/(name+'_reference_logits.npy'),logits)
        correct=sum(a==b for a,b in zip(p,y))
        row=dict(model=name,dataset=ds,input_resolution=res,correct=correct,test_count=len(y),accuracy_pct=100*correct/len(y),parameters=sum(z.numel() for z in model.parameters()),deployment_bytes=path.stat().st_size,training_checkpoint_bytes=(ROOT/'models'/(name+'.pth')).stat().st_size,median_ms=profile['median_ms'],p95_ms=profile['p95_ms'],observed_peak_inference_rss_bytes=profile['observed_peak_inference_rss_bytes'],rss_before_model_load_bytes=profile['rss_before_model_load_bytes'],os_peak_working_set_bytes=profile['os_peak_working_set_bytes'])
        rows.append(row);print(json.dumps(row),flush=True)
        predictions.extend(dict(model=name,dataset=ds,test_index=i,true_label=y[i],prediction=p[i],correct=int(y[i]==p[i])) for i in range(len(y)))
        latencies.extend(dict(model=name,repeat=i+1,latency_ms=t) for i,t in enumerate(profile['latency_samples_ms']))
    for filename,records in [('benchmark.csv',rows),('test_predictions.csv',predictions),('latency_samples.csv',latencies)]:
        with (ROOT/'results'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (ROOT/'results/benchmark.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'results/cpu_environment.json').write_text(json.dumps(dict(torch=torch.__version__,python=sys.version,platform=platform.platform(),processor=platform.processor(),threads=4,warmup=30,latency_repeats=200,memory_inference_runs=100,memory_sampling_seconds=.001),indent=2))
