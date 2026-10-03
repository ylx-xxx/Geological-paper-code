"""Prospective TrainData-only training; checkpoint selection uses ValidData only."""
import argparse
import csv
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train_l4s_qz.model import SwinUPerNetL4S, adapt_patch_embed_weight
from train_l4s_qz.losses import CombinedLoss
from .core import CHANNELS, NORMALIZATION, normalize, pairs, read_h5, sha256, confusion, metrics, write_json, verify_selection_split
from .evaluate import predict

class TrainingDataset(Dataset):
    """Cache normalized input; augmentation follows the historical ablation recipe."""
    def __init__(self, root, split, channels):
        if split not in ['train','val']:raise ValueError('Training cannot access test data')
        self.split=split;self.samples=pairs(root,split);self.cache=[]
        for ip,mp in self.samples:
            x=normalize(read_h5(ip,'img'))[CHANNELS[channels]]
            y=np.squeeze(read_h5(mp,'mask')).astype(np.int64)
            if not np.isin(y,[0,1,255]).all():raise ValueError(f'Invalid labels: {mp}')
            if x.shape[-2:]!=(128,128) or y.shape!=(128,128):raise ValueError('Expected native 128 x 128 patches')
            self.cache.append((x,y))
    def __len__(self):return len(self.cache)
    def __getitem__(self,i):
        x,y=self.cache[i]
        if self.split=='train':
            if random.random()<.5:x=x[:,:,::-1];y=y[:,::-1]
            if random.random()<.5:x=x[:,::-1,:];y=y[::-1,:]
            if random.random()<.5:
                k=random.choice([1,2,3]);x=np.rot90(x,k,(1,2));y=np.rot90(y,k)
            if random.random()<.3:x=x+np.random.normal(0,.02,x.shape).astype(np.float32)
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())

def resize_bias(source,target):
    """Bicubic interpolation of a square relative-position grid, per head."""
    old=int(source.shape[0]**.5);new=int(target.shape[0]**.5)
    if source.ndim!=2 or target.ndim!=2 or old*old!=source.shape[0] or new*new!=target.shape[0] or source.shape[1]!=target.shape[1]:
        raise ValueError('Unsupported relative-position bias shape')
    grid=source.T.reshape(1,source.shape[1],old,old).float()
    grid=torch.nn.functional.interpolate(grid,size=(new,new),mode='bicubic',align_corners=False)
    return grid.reshape(source.shape[1],new*new).T.to(dtype=target.dtype)

def transfer(model,path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    state=ckpt['model'] if 'model' in ckpt else ckpt
    target=model.state_dict();converted={};adapted=[]
    for key,value in state.items():
        key=key.removeprefix('module.')
        if key not in target:continue
        if value.shape==target[key].shape:converted[key]=value
        elif key.endswith('patch_embed.proj.weight'):
            converted[key]=adapt_patch_embed_weight(value,target[key]);adapted.append(key)
        elif key.endswith('relative_position_bias_table'):
            converted[key]=resize_bias(value,target[key]);adapted.append(key)
    allowed={'fuse.2.weight','fuse.2.bias'}
    missing=set(target)-set(converted)
    if missing-allowed:raise RuntimeError(f'Incomplete LoveDA transfer: {sorted(missing-allowed)}')
    model.load_state_dict(converted,strict=False)
    return {'direct_or_adapted_keys':len(converted),'adapted_keys':adapted,'new_classifier_keys':sorted(missing),'source_sha256':sha256(path)}

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--loveda-checkpoint',required=True)
    ap.add_argument('--channels',choices=list(CHANNELS),required=True)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--max-seconds',type=int,default=3300)
    args=ap.parse_args();verify_selection_split('val')
    if args.epochs<1 or args.batch_size<2 or args.max_seconds<1:raise ValueError('Invalid training limits')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Use a new run directory')
    if 'checkpoints' in out.parts:raise ValueError('Use a new runs directory; original checkpoint directories are protected')
    out.mkdir(parents=True)
    started=time.monotonic();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);torch.set_float32_matmul_precision('high')
    config=vars(args)|{'selection_split':'val','selection_metric':'Landslide_IoU','normalization':NORMALIZATION,
        'channel_indices':CHANNELS[args.channels],'lr':3e-5,'min_lr':1e-6,'weight_decay':.05,
        'label_smoothing':.02,'loss':'weighted_ce_plus_weighted_dice','patch_init':'positional_source_mean_extra',
        'val_tta_views':3,'source_checkpoint_sha256':sha256(args.loveda_checkpoint),
        'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'torch':torch.__version__,'cudnn_benchmark':False,'note':'new prospective run, not a replacement for historical scores'}
    write_json(out/'config.json',config)
    train=TrainingDataset(args.data_root,'train',args.channels);val=TrainingDataset(args.data_root,'val',args.channels)
    counts=np.array([sum(int((y==c).sum()) for _,y in train.cache) for c in [0,1]],dtype=np.float64)
    weights=1/np.sqrt(counts/counts.sum()+1e-12);weights=np.clip(weights/weights.mean(),.5,2.5)
    config['class_weights']=weights.tolist();config['train_samples']=len(train);config['val_samples']=len(val)
    write_json(out/'config.json',config)
    # Save sample identity without reading TestData.
    for name,ds in [('train',train),('val',val)]:
        write_json(out/f'{name}_manifest.json',[{'sample':ip.name,'image_sha256':sha256(ip),'mask_sha256':sha256(mp)} for ip,mp in ds.samples])
    gen=torch.Generator().manual_seed(args.seed)
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=(ds is train),num_workers=args.workers,
            drop_last=False,pin_memory=True,worker_init_fn=seed_worker,generator=gen if ds is train else None,
            persistent_workers=args.workers>0) for ds in [train,val]]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SwinUPerNetL4S(in_chans=len(CHANNELS[args.channels]),img_size=128)
    write_json(out/'transfer.json',transfer(model,args.loveda_checkpoint))
    model=model.to(device,memory_format=torch.channels_last)
    criterion=CombinedLoss(class_weights=weights.tolist(),label_smoothing=.02).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    best=-1.;best_epoch=0;done=0;reason='completed'
    for epoch in range(1,args.epochs+1):
        if time.monotonic()-started>=args.max_seconds:reason='time_limit';break
        model.train();loss_sum=0.;batches=0;timed_out=False
        for x,y in loaders[0]:
            if time.monotonic()-started>=args.max_seconds:timed_out=True;break
            x=x.to(device,memory_format=torch.channels_last);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):loss=criterion(model(x),y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
            loss_sum+=loss.item();batches+=1
        if timed_out:reason='time_limit_partial_epoch';break
        model.eval();cm=np.zeros((2,2),dtype=np.int64)
        with torch.inference_mode():
            for x,y in loaders[1]:
                x=x.to(device,memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):p=predict(model,x,True).argmax(1).cpu().numpy()
                cm+=confusion(p,y.numpy())
        score=metrics(cm);scheduler.step();done=epoch
        row={'epoch':epoch,'train_loss':loss_sum/max(batches,1),'val_landslide_iou':score['Landslide_IoU'],
             'val_landslide_f1':score['Landslide_F1'],'val_precision':score['Landslide_Precision'],
             'val_recall':score['Landslide_Recall'],'seconds':time.monotonic()-started}
        with (out/'history.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if epoch==1:w.writeheader()
            w.writerow(row)
        if score['Landslide_IoU']>best:
            best=score['Landslide_IoU'];best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'config':config,'selection_split':'val',
                        'selection_metric':'Landslide_IoU','validation_metrics':score},out/'best.pt')
        print(row,flush=True)
    summary={'completed_epochs':done,'planned_epochs':args.epochs,'status':reason,
             'best_validation_iou':best,'best_epoch':best_epoch,'selection_split':'val',
             'seconds':time.monotonic()-started,'test_evaluated':False}
    if (out/'best.pt').exists():summary['checkpoint_sha256']=sha256(out/'best.pt')
    write_json(out/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
