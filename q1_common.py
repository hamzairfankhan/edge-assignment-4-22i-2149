from pathlib import Path
import json, random, time, os, platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR100
from architectures import Classifier
ROOT=Path(__file__).resolve().parent; MODELS=ROOT/'models'; RESULTS=ROOT/'results'
SEED=2149; RESOLUTION=64
def setup(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def dataset(device):
    root=os.environ.get('CIFAR100_ROOT',str(ROOT/'data'))
    tr=CIFAR100(root,train=True,download=True); te=CIFAR100(root,train=False,download=True)
    x=torch.from_numpy(tr.data.transpose(0,3,1,2).copy()).to(device)
    y=torch.tensor(tr.targets,device=device)
    xt=torch.from_numpy(te.data.transpose(0,3,1,2).copy()).to(device); yt=torch.tensor(te.targets,device=device)
    order=torch.randperm(50000,generator=torch.Generator().manual_seed(SEED)).to(device)
    return x[order[:45000]],y[order[:45000]],x[order[45000:]],y[order[45000:]],xt,yt,tr.classes
def preprocess(raw,flips=None):
    x=raw.float()/255
    if flips is not None:
        x=x.clone(); x[flips]=x[flips].flip(3)
    x=F.interpolate(x,(RESOLUTION,RESOLUTION),mode='bilinear',align_corners=False)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (x-mean)/std
@torch.inference_mode()
def evaluate(m,x,y):
    m.eval(); pp=[]; loss=0.
    for start in range(0,len(y),256):
        z,_=m(preprocess(x[start:start+256])); pp.append(z.argmax(1)); loss+=F.cross_entropy(z,y[start:start+256],reduction='sum').item()
    p=torch.cat(pp); return dict(accuracy_pct=100*int((p==y).sum())/len(y),correct=int((p==y).sum()),test_count=len(y),loss=loss/len(y)),p.cpu().tolist()
def train(model,x,y,xv,yv,epochs,label,loss_fn=None,cache=None):
    params=list(model.parameters())+(list(loss_fn.parameters()) if loss_fn is not None else [])
    if model.name=='squeezenet':
        # The unnormalized Fire backbone needs a gentler fine-tuning rate.
        groups=[dict(params=model.network.features.parameters(),lr=.0001),dict(params=model.network.classifier.parameters(),lr=.001)]
        if loss_fn is not None:groups.append(dict(params=loss_fn.parameters(),lr=.001))
        opt=torch.optim.SGD(groups,lr=.001,momentum=.9,weight_decay=1e-4)
    else:opt=torch.optim.SGD(params,lr=.01,momentum=.9,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=x.is_cuda)
    history=[]; best=-1.; best_state=None; best_adapter=None
    # Reset each student training stream, so baseline/KD use matching batches/flips.
    torch.manual_seed(SEED)
    for epoch in range(epochs):
        start=time.perf_counter(); model.train(); totals=[0.,0.,0.,0.]; correct=0
        if loss_fn is not None: loss_fn.train()
        for ids in torch.randperm(len(y),device=x.device).split(128):
            flips=torch.rand(len(ids),device=x.device)<.5; xb=preprocess(x[ids],flips)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=x.device.type,dtype=torch.float16,enabled=x.is_cuda):
                logits,features=model(xb)
                if loss_fn is None:
                    loss=F.cross_entropy(logits,y[ids]); hard=loss; soft=loss.new_tensor(0.); feature=loss.new_tensor(0.)
                else:
                    view=flips.long(); tl=cache['logits'][view,ids].float(); tf=cache['features'][view,ids].float()
                    loss,hard,soft,feature=loss_fn(logits.float(),features.float(),y[ids],tl,tf)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            for k,z in enumerate([loss,hard,soft,feature]): totals[k]+=z.item()*len(ids)
            correct+=int((logits.argmax(1)==y[ids]).sum())
        scheduler.step(); val,_=evaluate(model,xv,yv)
        row=dict(stage=label,epoch=epoch+1,loss=totals[0]/len(y),hard_loss=totals[1]/len(y),soft_loss=totals[2]/len(y),feature_loss=totals[3]/len(y),train_accuracy_pct=100*correct/len(y),validation_accuracy_pct=val['accuracy_pct'],seconds=time.perf_counter()-start)
        history.append(row); print(json.dumps(row),flush=True)
        if val['accuracy_pct']>best:
            best=val['accuracy_pct']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if loss_fn is not None: best_adapter={k:v.detach().cpu().clone() for k,v in loss_fn.state_dict().items()}
        (RESULTS/(label+'_history.json')).write_text(json.dumps(history,indent=2))
    model.load_state_dict(best_state)
    if best_adapter is not None: loss_fn.load_state_dict(best_adapter)
    return history
def save(model,label,history,adapter=None):
    ck=dict(architecture=model.name,model={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=SEED,resolution=RESOLUTION,classes=100,initialization='ImageNet pretrained; CIFAR-100 classifier replaced',best_validation_accuracy=max(r['validation_accuracy_pct'] for r in history))
    if adapter is not None: ck['distillation_adapter']={k:v.detach().cpu() for k,v in adapter.state_dict().items()}
    torch.save(ck,MODELS/(label+'.pth'))
@torch.inference_mode()
def teacher_cache(teacher,x):
    # Match each student's exact original/flipped view, without repeated teacher passes.
    teacher.eval(); logits=[]; features=[]
    for flip in [False,True]:
        ll=[]; ff=[]
        for start in range(0,len(x),256):
            raw=x[start:start+256]; flags=torch.full((len(raw),),flip,device=x.device,dtype=torch.bool)
            l,f=teacher(preprocess(raw,flags)); ll.append(l.half()); ff.append(f.half())
        logits.append(torch.cat(ll)); features.append(torch.cat(ff))
    return dict(logits=torch.stack(logits),features=torch.stack(features))
