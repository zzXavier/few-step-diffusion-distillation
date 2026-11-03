import argparse
from pathlib import Path
import torch
from src.runner import demo,evaluate,reproduce

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["demo","reproduce","evaluate"])
    p.add_argument("--base-path",type=Path)
    p.add_argument("--adapter-path",type=Path)
    p.add_argument("--cache",type=Path,default=Path(".cache"))
    p.add_argument("--out",type=Path)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args()
    out=a.out or Path("scratch/demo" if a.action=="demo" else "results")
    if a.action=="evaluate":evaluate(out)
    else:
        torch.set_num_threads(2)
        (demo if a.action=="demo" else reproduce)(a.cache,out,torch.device(a.device),a.base_path,a.adapter_path)
