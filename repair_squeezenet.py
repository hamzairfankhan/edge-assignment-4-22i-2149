"""Re-run both SqueezeNet comparisons with the corrected raw-logit classifier.

The initial shared-LR runs collapsed to chance. Preserve them as a pilot record,
then replace the two final checkpoints/results with matched corrected runs.
"""
from q1_common import *
from architectures import load_classifier
from distillation import DistillationLoss
import gc
if __name__=='__main__':
    device=setup();x,y,xv,yv,xt,yt,classes=dataset(device)
    bundle=json.loads((RESULTS/'q1_results.json').read_text())
    pilot=dict(summaries=[r for r in bundle['summaries'] if r['model'].startswith('squeezenet')],histories={k:v for k,v in bundle['histories'].items() if k.startswith('squeezenet')},reason='Initial shared LR and clamped classifier collapsed to chance; final runs use a raw-logit head, official classifier initialization scale, and architecture-specific lower learning rates.')
    (RESULTS/'squeezenet_initial_pilot.json').write_text(json.dumps(pilot,indent=2))
    teacher=load_classifier(MODELS/'teacher_resnet50.pth').to(device);cache=teacher_cache(teacher,x);del teacher
    for mode in ['baseline','kd']:
        setup();name='squeezenet_'+mode;model=Classifier('squeezenet',pretrained=True).to(device)
        criterion=DistillationLoss(model.feature_dim).to(device) if mode=='kd' else None
        history=train(model,x,y,xv,yv,10,name,criterion,cache if criterion is not None else None)
        save(model,name,history,criterion);metric,p=evaluate(model,xt,yt);metric.update(model=name,dataset='CIFAR-100')
        bundle['summaries']=[metric if r['model']==name else r for r in bundle['summaries']];bundle['histories'][name]=history;bundle['predictions'][name]=p
        bundle['environment']['squeezenet_protocol']=dict(backbone_lr=.0001,classifier_lr=.001,adapter_lr=.001,head='raw 100-class convolution logits; terminal classifier ReLU removed',classifier_weight_init_std=.01,epochs=10)
        (RESULTS/'q1_results.json').write_text(json.dumps(bundle));print('CORRECTED SQUEEZENET TEST:',metric,flush=True)
        del model,criterion;gc.collect();torch.cuda.empty_cache()
