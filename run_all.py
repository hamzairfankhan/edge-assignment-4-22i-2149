"""Run Q1 training, Q2 evolutionary search, benchmark, ONNX export and checks."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for script in ['train_q1.py','nas_search.py','benchmark.py','export_onnx.py','verify.py']:
        print('RUN',script,flush=True);subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
