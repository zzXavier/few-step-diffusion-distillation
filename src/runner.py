import csv
import json
from pathlib import Path
import time

from diffusers import DDIMScheduler, DDPMScheduler, LCMScheduler
import numpy as np
from PIL import Image, ImageDraw
import torch

from . import sd
from .lcm_math import DDIMSolver, get_predicted_original_sample, scalings_for_boundary_conditions

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/protocol.json").read_text())
ADAPTER="latent-consistency/lcm-lora-sdv1-5"
REV="cf2fced511dbe7e26c8d1d397e728fbab875db4b"


def distill(pipe,out):
    import skimage
    data=Path(skimage.__file__).parent/"data"
    prompts=CFG["training_images"]
    base=pipe.unet
    trainable=sd.add_adapter(base,CFG["rank"])
    saved=sd.trainable_state(base)
    base.enable_gradient_checkpointing()
    base.train()
    optimizer=torch.optim.AdamW(trainable,lr=1.e-4)
    schedule=DDPMScheduler.from_config(pipe.scheduler.config)
    solver=DDIMSolver(schedule.alphas_cumprod.numpy()).to(pipe.device)
    alpha=schedule.alphas_cumprod.sqrt().to(pipe.device)
    sigma=(1-schedule.alphas_cumprod).sqrt().to(pipe.device)
    zs=sd.latents(pipe,[data/f for f in prompts],CFG["resolution"])
    context=sd.encode(pipe,list(prompts.values()))
    empty=sd.encode(pipe,[""])
    trace=[]
    for it in range(CFG["updates"]):
        i=it%len(prompts)
        z=zs[i:i+1]
        c=context[i:i+1]
        idx=torch.randint(0,50,(1,),device=pipe.device)
        t=solver.ddim_timesteps[idx]
        prev=(t-20).clamp_min(0)
        noisy=schedule.add_noise(z,torch.randn_like(z),t)
        def eps(x,ts,condition):
            return base(x.to(base.dtype),ts,encoder_hidden_states=condition).sample.float()
        def x0(x,ts,e):
            return get_predicted_original_sample(e,ts,x.float(),"epsilon",alpha,sigma)
        def consistency(x,ts,e):
            skip,scale=scalings_for_boundary_conditions(ts.float())
            return skip*x.float()+scale*x0(x,ts,e)
        # One frozen teacher DDIM step. Both branches see identical latents and text.
        with torch.no_grad():
            with sd.no_adapter(base):
                uncond=eps(noisy,t,empty)
                cond=eps(noisy,t,c)
                guided=uncond+5.0*(cond-uncond)
                previous=solver.ddim_step(x0(noisy,t,guided),guided,idx)
            target=consistency(previous,prev,eps(previous,prev,c))
        pred=consistency(noisy,t,eps(noisy,t,c))
        loss=torch.nn.functional.smooth_l1_loss(pred,target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step()
        trace.append(float(loss.detach()))
    update=sd.save_reload(base,saved,out/"trained.safetensors")
    base.eval()
    base.disable_gradient_checkpointing()
    (out/"training.json").write_text(json.dumps({"loss":trace,"adapter_update_l2":update,"updates":len(trace)},indent=2))


def sample(pipe,prompt,seed,steps,size,guidance=1.0):
    device=pipe.device
    sd.sync(device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    with torch.inference_mode():
        image=pipe(prompt,num_inference_steps=steps,height=size,width=size,guidance_scale=guidance,
                  generator=torch.Generator(device=device).manual_seed(seed)).images[0]
    sd.sync(device)
    elapsed=time.perf_counter()-start
    mem=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
    return image,elapsed,mem


def release_pipeline(device):
    import gc
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()


def reproduce(cache,out,device,base_path,adapter_path):
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    sd.seed_all(CFG["seed"])
    out.mkdir(parents=True,exist_ok=True)
    pipe=sd.load_pipeline(device,False,cache,base_path)
    distill(pipe,out)
    del pipe
    release_pipeline(device)
    images=out/"predictions"
    images.mkdir(exist_ok=True)
    records=[]
    for method in ("teacher","trained","public"):
        pipe=sd.load_pipeline(device,False,cache,base_path)
        schedule_cfg=pipe.scheduler.config
        if method=="teacher":
            pipe.scheduler=DDIMScheduler.from_config(schedule_cfg)
        else:
            pipe.scheduler=LCMScheduler.from_config(schedule_cfg)
            if method=="trained":
                sd.add_adapter(pipe.unet,CFG["rank"])
                set_peft_model_state_dict(pipe.unet,load_file(str(out/"trained.safetensors")))
                pipe.unet.eval()
            else:
                pipe.load_lora_weights(str(adapter_path) if adapter_path else ADAPTER,
                    revision=None if adapter_path else REV, cache_dir=str(cache/"hub"),
                    weight_name="pytorch_lora_weights.safetensors")
        sample(pipe,"a grey cube",37,4,CFG["resolution"])
        for j,prompt in enumerate(CFG["evaluation_prompts"]):
            for seed in CFG["seeds"]:
                for steps in CFG["steps"]:
                    img,secs,mem=sample(pipe,prompt,seed,steps,CFG["resolution"])
                    path=f"{method}-p{j}-s{seed}-n{steps}.png"
                    img.save(images/path)
                    records.append(dict(method=method,prompt=j,seed=seed,steps=steps,file=path,seconds=secs,peak_cuda_bytes=mem))
        del pipe
        release_pipeline(device)
    (out/"measurements.json").write_text(json.dumps(records,indent=2))
    evaluate(out)


def image_error(a,b):
    return float(np.mean((np.asarray(a,dtype="f")/255.-np.asarray(b,dtype="f")/255.)**2))


def evaluate(out):
    logs=json.loads((out/"measurements.json").read_text())
    directory=out/"predictions"
    expected={(m,i,s,n) for m in ("teacher","trained","public") for i in range(len(CFG["evaluation_prompts"])) for s in CFG["seeds"] for n in CFG["steps"]}
    if len(logs)!=len(expected) or {(a["method"],a["prompt"],a["seed"],a["steps"]) for a in logs}!=expected:raise ValueError("Incomplete paired inference grid")
    for row in logs:
        with Image.open(directory/row["file"]) as pred, Image.open(directory/f"teacher-p{row['prompt']}-s{row['seed']}-n20.png") as target:
            row["mse_to_teacher20"]=image_error(pred,target)
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["method","steps","cases","mean_mse_to_teacher20","p50_seconds","p95_seconds","max_cuda_bytes"])
        for method in ("teacher","trained","public"):
            for n in CFG["steps"]:
                rows=[r for r in logs if r["method"]==method and r["steps"]==n]
                peaks=[r["peak_cuda_bytes"] for r in rows if r["peak_cuda_bytes"] is not None]
                w.writerow([method,n,len(rows),np.mean([r["mse_to_teacher20"] for r in rows]),*np.percentile([r["seconds"] for r in rows],[50,95]),max(peaks,default="")])
    canvas=Image.new("RGB",(3*160,3*180),"white")
    for y,method in enumerate(("teacher","trained","public")):
        for x,n in enumerate(CFG["steps"]):
            with Image.open(directory/f"{method}-p0-s{CFG['seeds'][0]}-n{n}.png") as im:
                canvas.paste(im.resize((160,160)),(160*x,180*y+20))
            ImageDraw.Draw(canvas).text((160*x+2,180*y+1),f"{method}, {n} steps",fill="black")
    canvas.save(out/"comparison.png")
    print((out/"summary.csv").read_text())


def demo(cache,out,device,base_path,adapter_path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    out.mkdir(parents=True,exist_ok=True)
    panel=Image.new("RGB",(3*256,280),"white")
    modes=[("teacher",20,7.5),("trained",4,1.0),("public",4,1.0)]
    details=[]
    for idx,(method,n,guidance) in enumerate(modes):
        pipe=sd.load_pipeline(device,False,cache,base_path)
        cfg=pipe.scheduler.config
        pipe.scheduler=(DDIMScheduler if method=="teacher" else LCMScheduler).from_config(cfg)
        if method=="trained":
            sd.add_adapter(pipe.unet,CFG["rank"])
            set_peft_model_state_dict(pipe.unet,load_file(str(ROOT/"results/trained.safetensors")))
            pipe.unet.eval()
        elif method=="public":
            pipe.load_lora_weights(str(adapter_path) if adapter_path else ADAPTER,revision=None if adapter_path else REV,
                                   weight_name="pytorch_lora_weights.safetensors",cache_dir=str(cache/"hub"))
        photo,secs,_=sample(pipe,CFG["evaluation_prompts"][0],CFG["seeds"][0],n,512,guidance)
        photo.save(out/f"{method}.png")
        panel.paste(photo.resize((256,256)),(idx*256,24))
        ImageDraw.Draw(panel).text((idx*256+5,3),f"{method}: {n} steps, CFG {guidance}",fill="black")
        details.append({"method":method,"steps":n,"guidance":guidance,"seconds":secs,"resolution":512})
        del pipe
        release_pipeline(device)
    panel.save(out/"native-preview.png")
    (out/"native-settings.json").write_text(json.dumps(details,indent=2))
    print(out/"native-preview.png")
