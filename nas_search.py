"""Elitist evolutionary NAS, measured constraints, fresh final retraining."""
from pathlib import Path
import os,json,csv,subprocess,sys,random,time,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from nas_space import SearchCNN,sample_config,mutate
ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results';MODELS=ROOT/'models';CANDIDATES=ROOT/'nas_candidates'
SEED=2149;MAX_LATENCY_MS=1.2;MAX_STORAGE_BYTES=180*1024
def setup(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    for p in [RESULTS,MODELS,CANDIDATES]:p.mkdir(exist_ok=True)
def data():
    path=os.environ.get('CIFAR10_ROOT',str(ROOT/'data'))
    train=CIFAR10(path,train=True,download=True);test=CIFAR10(path,train=False,download=True)
    mean=torch.tensor([.4914,.4822,.4465]).view(1,3,1,1);std=torch.tensor([.247,.2435,.2616]).view(1,3,1,1)
    def tensor(d):return (torch.from_numpy(d.data.transpose(0,3,1,2).copy()).float()/255-mean)/std,torch.tensor(d.targets)
    x,y=tensor(train);xt,yt=tensor(test);order=torch.randperm(50000,generator=torch.Generator().manual_seed(SEED))
    return x[order[:45000]],y[order[:45000]],x[order[45000:]],y[order[45000:]],xt,yt
def resize(x,res):return x if res==32 else F.interpolate(x,(res,res),mode='bilinear',align_corners=False)
@torch.inference_mode()
def evaluate(model,x,y,res):
    model.eval();pp=[]
    for start in range(0,len(y),256):pp.append(model(resize(x[start:start+256],res)).argmax(1))
    p=torch.cat(pp);return 100*int((p==y).sum())/len(y),p.tolist()
def train(model,config,x,y,xv,yv,epochs,label):
    optimizer=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs);history=[]
    for epoch in range(epochs):
        model.train();correct=0;total=0.;start=time.perf_counter()
        for ids in torch.randperm(len(y)).split(256):
            xb=x[ids].clone();flip=torch.rand(len(ids))<.5;xb[flip]=xb[flip].flip(3);xb=resize(xb,config['resolution'])
            optimizer.zero_grad();z=model(xb);loss=F.cross_entropy(z,y[ids]);loss.backward();optimizer.step();total+=loss.item()*len(ids);correct+=int((z.argmax(1)==y[ids]).sum())
        scheduler.step();val,_=evaluate(model,xv,yv,config['resolution'])
        row=dict(stage=label,epoch=epoch+1,train_loss=total/len(y),train_accuracy_pct=100*correct/len(y),validation_accuracy_pct=val,seconds=time.perf_counter()-start);history.append(row);print(json.dumps(row),flush=True)
    return history
def save_executable(model,config,path):
    torch.jit.trace(model.eval(),torch.zeros(1,3,config['resolution'],config['resolution'])).save(str(path))
def measure(path,config):
    text=subprocess.check_output([sys.executable,str(ROOT/'cpu_probe.py'),str(path),'--resolution',str(config['resolution'])],text=True)
    return json.loads(text)
def feasible(profile):return profile['median_ms']<=MAX_LATENCY_MS and profile['checkpoint_bytes']<=MAX_STORAGE_BYTES
def write_history(history):
    (RESULTS/'nas_search_history.json').write_text(json.dumps(history,indent=2))
    records=[]
    for row in history:
        p=row['profile'];c=row['config'];records.append(dict(trial=row['trial'],generation=row['generation'],feasible=row['feasible'],rejection=row['rejection'],validation_accuracy_pct=row.get('validation_accuracy_pct'),depth=c['depth'],channels=json.dumps(c['channels']),kernels=json.dumps(c['kernels']),blocks=json.dumps(c['blocks']),activation=c['activation'],resolution=c['resolution'],parameters=row['parameters'],checkpoint_bytes=p['checkpoint_bytes'],median_ms=p['median_ms'],p95_ms=p['p95_ms'],observed_peak_inference_rss_bytes=p['observed_peak_inference_rss_bytes']))
    with (RESULTS/'nas_search_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
if __name__=='__main__':
    setup();x,y,xv,yv,xt,yt=data();rng=random.Random(SEED);history=[];seen=set()
    population=[dict(depth=4,channels=[64,96,128,192],kernels=[5]*4,blocks=['standard']*4,activation='hardswish',resolution=40),dict(depth=2,channels=[16,24],kernels=[3,3],blocks=['standard','standard'],activation='relu',resolution=32),dict(depth=3,channels=[24,48,64],kernels=[3,3,3],blocks=['depthwise']*3,activation='relu6',resolution=32),dict(depth=4,channels=[16,24,48,64],kernels=[3,5,3,5],blocks=['standard','depthwise','depthwise','depthwise'],activation='hardswish',resolution=24),sample_config(rng),sample_config(rng)]
    for generation in range(3):
        for config in population:
            key=json.dumps(config,sort_keys=True)
            if key in seen:continue
            seen.add(key);trial=len(history);setup(SEED+trial);model=SearchCNN(config);path=CANDIDATES/f'trial_{trial:02}.pt';save_executable(model,config,path)
            profile=measure(path,config);row=dict(trial=trial,generation=generation,config=config,parameters=sum(p.numel() for p in model.parameters()),profile=profile,feasible=feasible(profile),rejection='')
            if row['feasible']:
                hist=train(model,config,x,y,xv,yv,2,f'nas_trial_{trial:02}');row['training_history']=hist;row['validation_accuracy_pct']=hist[-1]['validation_accuracy_pct']
                save_executable(model,config,path);row['profile']=measure(path,config);row['feasible']=feasible(row['profile'])
                if not row['feasible']:row['rejection']='Post-training measured constraint violation'
            else:row['rejection']='Pre-training measured constraint violation'
            history.append(row);write_history(history);print('NAS CANDIDATE',json.dumps({k:v for k,v in row.items() if k not in ['profile','training_history']}),flush=True)
        parents=sorted([r for r in history if r['feasible']],key=lambda r:r['validation_accuracy_pct'],reverse=True)[:2]
        if not parents:raise RuntimeError('No feasible candidate found under declared constraints')
        population=[]
        while len(population)<6:
            child=mutate(rng.choice(parents)['config'],rng);key=json.dumps(child,sort_keys=True)
            if key not in seen and all(json.dumps(c,sort_keys=True)!=key for c in population):population.append(child)
    best=max((r for r in history if r['feasible']),key=lambda r:r['validation_accuracy_pct'])
    chosen=dict(config=best['config'],selected_trial=best['trial'],search_validation_accuracy_pct=best['validation_accuracy_pct'],constraints=dict(max_median_latency_ms=MAX_LATENCY_MS,max_checkpoint_bytes=MAX_STORAGE_BYTES),selection='Highest validation accuracy among measured feasible candidates; no test-set selection',fresh_retraining_seed=SEED)
    (RESULTS/'best_architecture.json').write_text(json.dumps(chosen,indent=2));setup();model=SearchCNN(best['config'])
    init_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    hist=train(model,best['config'],x,y,xv,yv,15,'nas_final_from_scratch')
    torch.save(dict(config=best['config'],model=model.state_dict(),seed=SEED,initialization='fresh random weights; candidate checkpoint not loaded',initial_parameter_sha256=init_hash),MODELS/'nas_best.pth');save_executable(model,best['config'],MODELS/'nas_best.pt')
    profile=measure(MODELS/'nas_best.pt',best['config']);assert feasible(profile),'Retrained model violates measured constraints'
    acc,pred=evaluate(model,xt,yt,best['config']['resolution'])
    final=dict(model='nas_best',dataset='CIFAR-10',accuracy_pct=acc,correct=sum(a==b for a,b in zip(pred,yt.tolist())),test_count=10000,parameters=sum(p.numel() for p in model.parameters()),profile=profile,training_history=hist,config=best['config'],initial_parameter_sha256=init_hash)
    (RESULTS/'nas_final.json').write_text(json.dumps(final,indent=2));(RESULTS/'nas_test_predictions.json').write_text(json.dumps(dict(predictions=pred,true_labels=yt.tolist())))
    print('NAS COMPLETE',acc,'parameters',final['parameters'],'latency',profile['median_ms'],'size',profile['checkpoint_bytes'],flush=True)
