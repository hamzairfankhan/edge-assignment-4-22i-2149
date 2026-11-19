"""Restore four checkpoints lost when the November 15 Colab session expired.
The teacher and two baseline checkpoints remain unchanged. All new results and
hashes replace only their corresponding entries; the earlier records are retained.
"""
from q1_common import *
from architectures import load_classifier
from distillation import DistillationLoss
import gc,hashlib,sys,timm,torchvision
if __name__=='__main__':
    device=setup();x,y,xv,yv,xt,yt,classes=dataset(device)
    bundle=json.loads((RESULTS/'q1_results.json').read_text())
    (RESULTS/'q1_nov15_record.json').write_text(json.dumps(bundle))
    teacher=load_classifier(MODELS/'teacher_resnet50.pth').to(device);cache=teacher_cache(teacher,x);del teacher
    for architecture,mode in [('mobilenet_v2','kd'),('squeezenet','kd'),('efficientnet_lite0','baseline'),('efficientnet_lite0','kd')]:
        setup();name=architecture+'_'+mode;model=Classifier(architecture,pretrained=True).to(device)
        criterion=DistillationLoss(model.feature_dim).to(device) if mode=='kd' else None
        history=train(model,x,y,xv,yv,10,name,criterion,cache if criterion is not None else None)
        save(model,name,history,criterion);metric,p=evaluate(model,xt,yt);metric.update(model=name,dataset='CIFAR-100')
        bundle['summaries']=[metric if r['model']==name else r for r in bundle['summaries']];bundle['histories'][name]=history;bundle['predictions'][name]=p
        bundle.setdefault('checkpoint_recovery',{})[name]='Retrained 19 November 2026 after original runtime expired before transfer; same teacher/split/seed/budget.'
        bundle['model_sha256'][name+'.pth']=hashlib.sha256((MODELS/(name+'.pth')).read_bytes()).hexdigest()
        bundle['environment'].update(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,python=sys.version,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        (RESULTS/'q1_results.json').write_text(json.dumps(bundle));print('RESTORED TEST:',metric,flush=True)
        del model,criterion;gc.collect();torch.cuda.empty_cache()
    print('CHECKPOINT RECOVERY COMPLETE',flush=True)
