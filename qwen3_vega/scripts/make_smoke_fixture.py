"""Create tiny, independent synthetic media for a public checkout smoke test."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);root=p.parse_args().output
root.mkdir(parents=True,exist_ok=False)
for split,n in (("train",3),("tune",2)):
    rows=[]
    for i in range(n):
        name=f"{split}-{i}";frames=[]
        for j in range(8):
            f=np.full((128,160,3),30+i*30,dtype=np.uint8)
            f[20:60,10+j*4:40+j*4,0]=250
            frames.append(f)
        if i==0:
            path=root/name;path.mkdir()
            for j,f in enumerate(frames):Image.fromarray(f).save(path/f"{j:04d}.jpg")
            kind="frames"
        else:
            path=root/(name+".mp4");writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),2,(160,128))
            if not writer.isOpened():raise RuntimeError("Cannot create MPEG4 fixture")
            for f in frames:writer.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
            writer.release();kind="video"
        rows.append({"id":name,"path":path.name,"kind":kind,"type":"smoke","conversations":[
            {"from":"human","value":"<video>\nWhich way does the red square move?"},{"from":"gpt","value":"Right."}]})
    (root/(split+".json")).write_text(json.dumps(rows))
