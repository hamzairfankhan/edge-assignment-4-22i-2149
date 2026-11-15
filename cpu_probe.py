"""Isolated CPU inference probe. Reports process RSS, not activation-only memory."""
import argparse,json,time,threading
import numpy as np
import torch,psutil
from pathlib import Path
def probe(path,resolution,threads=4):
    torch.set_num_threads(threads); process=psutil.Process(); before=process.memory_info().rss
    m=torch.jit.load(str(path),map_location='cpu').eval(); x=torch.zeros(1,3,resolution,resolution)
    samples=[]
    with torch.inference_mode():
        for _ in range(30):m(x)
        for _ in range(200):
            t=time.perf_counter_ns();m(x);samples.append((time.perf_counter_ns()-t)/1e6)
    observed=[process.memory_info().rss];done=threading.Event()
    def sample_memory():
        while not done.is_set():
            observed.append(process.memory_info().rss);done.wait(.001)
    worker=threading.Thread(target=sample_memory);worker.start()
    with torch.inference_mode():
        for _ in range(100):m(x)
    done.set();worker.join();observed.append(process.memory_info().rss)
    info=process.memory_info()
    return dict(checkpoint_bytes=Path(path).stat().st_size,median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),rss_before_model_load_bytes=before,observed_peak_inference_rss_bytes=max(observed),rss_delta_vs_before_load_bytes=max(observed)-before,os_peak_working_set_bytes=getattr(info,'peak_wset',None),latency_samples_ms=samples,threads=threads,resolution=resolution)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--resolution',type=int,required=True);a=p.parse_args()
    print(json.dumps(probe(a.checkpoint,a.resolution)))
