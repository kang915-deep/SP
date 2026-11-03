"""Official VEGA 4--8 uniform frames, image chat, and shifted assistant CE labels."""
import hashlib,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, DECORDError
import cv2
from qwen_vl.data.utils import load_and_preprocess_images

def read_frames(row):
    path=Path(row['path'])
    if row['kind']=='frames':
        files=sorted(p for p in path.iterdir() if p.is_file())
        length=len(files);fps=1.0
    else:
        reader=VideoReader(str(path),num_threads=1)
        length=len(reader);fps=reader.get_avg_fps()
    n=min(max(round(length/fps/2),4),8)
    indices=np.unique(np.linspace(0,length-1,n,dtype=int))
    if row['kind']=='frames':images=[Image.open(files[i]).convert('RGB') for i in indices]
    else:
        try:
            images=[Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]
        except DECORDError as error:
            if 'Unable to handle EOF' not in str(error):raise
            # The observed EOF failure is random-seek related. Sequential decode
            # preserves the same indices; the failed fixture's first 7 frames
            # matched Decord pixel-for-pixel and all 8 indices were recovered.
            capture=cv2.VideoCapture(str(path));selected={};index=0
            wanted=set(indices.tolist())
            while index<=indices[-1]:
                ok,frame=capture.read()
                if not ok:break
                if index in wanted:selected[index]=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                index+=1
            capture.release()
            assert len(selected)==len(indices),('Original sampled frames unavailable',row['id'],indices.tolist(),list(selected))
            images=[selected[int(i)] for i in indices]
            print('DECORD_EOF_RECOVERED',row['id'],indices.tolist(),flush=True)
    # Same VEGA 518px preprocessing; only final grid changes 28 -> 32 for Qwen3.
    return load_and_preprocess_images(images)

def prepare(row,processor,frame_reader=read_frames):
    frames=frame_reader(row)
    unit=processor.image_processor.patch_size*processor.image_processor.merge_size
    h,w=frames.shape[-2:];frames=frames[:,:,:h//unit*unit,:w//unit*unit]
    vision=processor.image_processor(frames,return_tensors='pt',do_rescale=False,do_resize=False)
    grid=vision['image_grid_thw'];tokens=(grid.prod(-1)//processor.image_processor.merge_size**2).tolist()
    conv=[dict(c) for c in row['conversations']]
    # Replace exactly one video marker, preserving the original question/units/options.
    assert conv[0]['value'].count('<video>')==1,row['id']
    conv[0]['value']=conv[0]['value'].replace('<video>','<image>'*len(frames)).replace('<image>\n','<image>')
    tok=processor.tokenizer
    ids=tok.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n',add_special_tokens=False)
    labels=[-100]*len(ids);image_index=0
    for c in conv:
        role={'human':'user','gpt':'assistant'}[c['from']];content=c['value']
        if role=='user':
            parts=content.split('<image>');text=parts[0]
            for part in parts[1:]:
                text+='<|vision_start|>'+'<|image_pad|>'*tokens[image_index]+'<|vision_end|>'+part;image_index+=1
            content=text
        header=tok.encode('<|im_start|>'+role+'\n',add_special_tokens=False)
        body=tok.encode(content+'<|im_end|>\n',add_special_tokens=False)
        ids+=header+body
        labels+=([-100]*len(header)+body) if role=='assistant' else [-100]*(len(header)+len(body))
    assert image_index==len(grid) and len(ids)<=12800 and any(v!=-100 for v in labels),row['id']
    ids=torch.tensor([ids],dtype=torch.long)
    return {'input_ids':ids,'labels':torch.tensor([labels]),'attention_mask':torch.ones_like(ids),
            'mm_token_type_ids':(ids==tok.convert_tokens_to_ids('<|image_pad|>')).long(),
            'pixel_values':vision['pixel_values'],'image_grid_thw':grid,'wan_images':frames}

def sample_seed(qid,epoch,phase='train'):
    return int(hashlib.sha256(f'vega-qwen3-0:{phase}:{epoch}:{qid}'.encode()).hexdigest()[:15],16)
