"""Train the teacher and six independent student runs; test only final selected checkpoints."""
import argparse, gc
from q1_common import *
from distillation import DistillationLoss
from architectures import load_classifier
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--teacher-epochs',type=int,default=15);parser.add_argument('--student-epochs',type=int,default=10);a=parser.parse_args()
    device=setup(); print('TRAINING DEVICE:',device,torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),flush=True)
    x,y,xv,yv,xt,yt,classes=dataset(device)
    summaries=[]; histories={}; predictions={}
    if (MODELS/'teacher_resnet50.pth').exists():
        teacher=load_classifier(MODELS/'teacher_resnet50.pth').to(device)
        history=json.loads((RESULTS/'teacher_resnet50_history.json').read_text())
    else:
        teacher=Classifier('resnet50',pretrained=True).to(device)
        history=train(teacher,x,y,xv,yv,a.teacher_epochs,'teacher_resnet50');save(teacher,'teacher_resnet50',history)
    histories['teacher_resnet50']=history
    metric,p=evaluate(teacher,xt,yt);metric.update(model='teacher_resnet50',dataset='CIFAR-100');summaries.append(metric);predictions['teacher_resnet50']=p
    print('TEACHER TEST:',metric,flush=True)
    cache=teacher_cache(teacher,x);del teacher;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    for architecture in ['mobilenet_v2','squeezenet','efficientnet_lite0']:
        for mode in ['baseline','kd']:
            label=architecture+'_'+mode;setup()
            model=Classifier(architecture,pretrained=True).to(device)
            adapter=DistillationLoss(model.feature_dim).to(device) if mode=='kd' else None
            history=train(model,x,y,xv,yv,a.student_epochs,label,adapter,cache if adapter is not None else None)
            save(model,label,history,adapter); histories[label]=history
            metric,p=evaluate(model,xt,yt);metric.update(model=label,dataset='CIFAR-100');summaries.append(metric);predictions[label]=p
            print('FINAL TEST:',metric,flush=True)
            del model,adapter;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            bundle=dict(environment=dict(torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,platform=platform.platform(),seed=SEED,train_count=45000,validation_count=5000,test_count=10000,resolution=RESOLUTION,teacher_epochs=a.teacher_epochs,student_epochs=a.student_epochs,temperature=4,alpha=.5,beta=1),summaries=summaries,histories=histories,predictions=predictions,true_labels=yt.cpu().tolist(),classes=classes)
            (RESULTS/'q1_results.json').write_text(json.dumps(bundle))
    print('QUESTION 1 TRAINING COMPLETE',flush=True)
