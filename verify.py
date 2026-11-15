"""Independently check artifacts, predictions, raw timing summaries and NAS selection."""
import csv,json,hashlib
import torch,numpy as np,onnx
from inference import ROOT,NAMES,load_model
from architectures import load_classifier
from nas_space import SearchCNN
if __name__=='__main__':
    torch.set_num_threads(4);results=ROOT/'results';checks=[]
    benchmark=json.loads((results/'benchmark.json').read_text());pred=list(csv.DictReader((results/'test_predictions.csv').open()));times=list(csv.DictReader((results/'latency_samples.csv').open()))
    onnx_rows=json.loads((results/'onnx_validation.json').read_text());onnx_pred=list(csv.DictReader((results/'onnx_predictions.csv').open()))
    for name in NAMES:
        m,res,ds=load_model(name);columns=10 if ds=='CIFAR-10' else 100
        with torch.inference_mode():assert m(torch.zeros(2,3,res,res)).shape==(2,columns)
        script=torch.jit.load(str(ROOT/'models'/(name+'.pt')))
        with torch.inference_mode():assert script(torch.zeros(2,3,res,res)).shape==(2,columns)
        row=next(r for r in benchmark if r['model']==name);pp=[p for p in pred if p['model']==name];tt=[float(t['latency_ms']) for t in times if t['model']==name]
        assert len(pp)==10000 and len(tt)==200
        correct=sum(p['prediction']==p['true_label'] for p in pp)
        assert correct==row['correct'] and row['accuracy_pct']==correct/100
        assert abs(float(np.median(tt))-row['median_ms'])<1e-10
        assert abs(float(np.percentile(tt,95))-row['p95_ms'])<1e-10
        onnx.checker.check_model(onnx.load(str(ROOT/'models'/(name+'.onnx'))))
        op=[p for p in onnx_pred if p['model']==name];o=next(r for r in onnx_rows if r['model']==name)
        assert len(op)==10000 and sum(p['prediction']==p['true_label'] for p in op)==o['correct']
        assert o['max_absolute_logit_difference']<.01
        checks.append(name+': eager/TorchScript reload, test predictions, accuracy, latency samples and full-test ONNX validation pass')
    q1=json.loads((results/'q1_results.json').read_text())
    for r in q1['summaries']:
        pp=q1['predictions'][r['model']];yy=q1['true_labels'];assert len(pp)==10000 and sum(a==b for a,b in zip(pp,yy))==r['correct']
    teacher=load_classifier(ROOT/'models/teacher_resnet50.pth')
    assert sum(p.numel() for p in teacher.parameters())==23712932
    for name in ['mobilenet_v2','squeezenet','efficientnet_lite0']:
        a=torch.load(ROOT/'models'/(name+'_baseline.pth'),weights_only=True,map_location='cpu');b=torch.load(ROOT/'models'/(name+'_kd.pth'),weights_only=True,map_location='cpu')
        assert 'distillation_adapter' not in a and 'distillation_adapter' in b
        assert any(not torch.equal(a['model'][key],b['model'][key]) for key in a['model'] if a['model'][key].is_floating_point())
    history=json.loads((results/'nas_search_history.json').read_text());chosen=json.loads((results/'best_architecture.json').read_text());final=json.loads((results/'nas_final.json').read_text())
    feasible=[r for r in history if r['feasible']];selected=next(r for r in history if r['trial']==chosen['selected_trial'])
    assert selected['feasible'] and selected['validation_accuracy_pct']==max(r['validation_accuracy_pct'] for r in feasible)
    for r in history:
        valid=r['profile']['median_ms']<=chosen['constraints']['max_median_latency_ms'] and r['profile']['checkpoint_bytes']<=chosen['constraints']['max_checkpoint_bytes']
        assert valid==r['feasible']
    assert final['profile']['median_ms']<=chosen['constraints']['max_median_latency_ms'] and final['profile']['checkpoint_bytes']<=chosen['constraints']['max_checkpoint_bytes']
    assert final['config']==chosen['config'] and len(final['training_history'])==15
    ck=torch.load(ROOT/'models/nas_best.pth',weights_only=True,map_location='cpu');assert ck['initialization']=='fresh random weights; candidate checkpoint not loaded'
    torch.manual_seed(2149);fresh=SearchCNN(chosen['config'])
    initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in fresh.parameters())).hexdigest()
    assert initial_hash==ck['initial_parameter_sha256']==final['initial_parameter_sha256']
    checks.extend(['Teacher is the standard 23,712,932-parameter ResNet-50 with a 100-class head','Six distinct student checkpoints: hard-label baselines and KD checkpoints with training-only adapters','All Colab test prediction counts agree with saved summaries','NAS selects the highest-validation feasible candidate; every rejection follows measured limits','Selected NAS architecture is freshly retrained for 15 epochs and meets final measured limits'])
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').iterdir() if p.is_file()}
    report=dict(status='PASS',checks=checks,model_sha256=hashes,cpu_prediction_rows=len(pred),onnx_prediction_rows=len(onnx_pred),latency_rows=len(times),nas_candidates=len(history),nas_feasible=len(feasible))
    (results/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
