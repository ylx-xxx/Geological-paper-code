"""Evaluate a fixed checkpoint; never select/promote checkpoints from test scores."""
import argparse
import csv
import importlib.metadata
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from train_l4s_qz.model import SwinUPerNetL4S
from .core import CHANNELS, NORMALIZATION, normalize, pairs, read_h5, sha256, confusion, metrics, write_json

class EvaluationDataset(Dataset):
    def __init__(self, root, split, channels):
        self.samples = pairs(root, split)
        self.channels = CHANNELS[channels]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        image, mask = self.samples[i]
        x = normalize(read_h5(image, 'img'))[self.channels]
        y = np.squeeze(read_h5(mask, 'mask')).astype(np.int64)
        if x.shape[-2:] != (128,128) or y.shape != (128,128):
            raise ValueError('This protocol requires native 128 x 128 patches')
        if not np.isin(y, [0,1,255]).all():
            raise ValueError(f'Unexpected mask labels in {mask}')
        return torch.from_numpy(x), torch.from_numpy(y), i

def predict(model, x, tta):
    logits = model(x)
    if tta:
        h = torch.flip(model(torch.flip(x, [3])), [3])
        v = torch.flip(model(torch.flip(x, [2])), [2])
        logits = (logits+h+v)/3.
    return logits

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--expected-sha256', required=True, help='Freeze checkpoint identity before evaluation')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['val','test'], default='test')
    ap.add_argument('--channels', choices=list(CHANNELS), default='full14')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tta', type=int, choices=[0,1], default=1)
    ap.add_argument('--amp', type=int, choices=[0,1], default=1)
    ap.add_argument('--channels-last', type=int, choices=[0,1], default=1)
    ap.add_argument('--save-predictions', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Use a new output directory; previous evaluations are immutable')
    actual = sha256(args.checkpoint)
    if actual != args.expected_sha256:
        raise ValueError('Checkpoint SHA256 differs from the frozen identity')
    ds = EvaluationDataset(args.data_root, args.split, args.channels)
    model = SwinUPerNetL4S(in_chans=len(CHANNELS[args.channels]), img_size=128)
    # Only load checkpoints obtained from a trusted source.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    del ckpt, state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    provenance = {'checkpoint_sha256': actual, 'split': args.split,
        'channels': args.channels, 'channel_indices': CHANNELS[args.channels],
        'normalization': NORMALIZATION, 'tta_views': 3 if args.tta else 1,
        'batch_size': args.batch_size, 'amp': bool(args.amp), 'channels_last': bool(args.channels_last),
        'checkpoint_load': 'strict', 'sample_count': len(ds), 'purpose': 'fixed-checkpoint evaluation, no selection',
        'versions': {n:importlib.metadata.version(n) for n in ['torch','torchvision','timm','numpy','h5py']},
        'source_sha256': {str(p.relative_to(source_root)):sha256(p) for p in
            [Path(__file__),Path(__file__).with_name('core.py'),source_root/'train_l4s_qz/model.py']}}
    write_json(out/'protocol.json', provenance)
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                        pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    rows, masks_out, names = [], [], []
    total = np.zeros((2,2),dtype=np.int64)
    started = time.monotonic()
    with torch.inference_mode():
        for x,y,indices in loader:
            x=x.to(device)
            if args.channels_last:x=x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type,enabled=bool(args.amp) and device.type=='cuda'):
                pred=predict(model,x,bool(args.tta)).argmax(1).cpu().numpy().astype(np.uint8)
            for p,g,i in zip(pred,y.numpy(),indices.tolist()):
                cm=confusion(p,g);total+=cm
                ip,mp=ds.samples[i]
                row={'sample':ip.name,'image_sha256':sha256(ip),'mask_sha256':sha256(mp),
                     'positive_pixels':int((g==1).sum()),'valid_pixels':int((g!=255).sum()),**metrics(cm)}
                # Empty-ground-truth patches must not enter foreground patch means.
                row['foreground_patch_iou']=row['Landslide_IoU'] if row['positive_pixels'] else ''
                rows.append(row)
                if args.save_predictions:masks_out.append(p);names.append(ip.name)
            print(f'Evaluated {len(rows)}/{len(ds)}',flush=True)
    summed=np.array([[sum(r['TN'] for r in rows),sum(r['FP'] for r in rows)],
                     [sum(r['FN'] for r in rows),sum(r['TP'] for r in rows)]])
    if not np.array_equal(summed,total):raise AssertionError('Sample/global counts differ')
    with (out/'per_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={**metrics(total),'samples':len(rows),'confusion_matrix':total.tolist(),
             'sample_global_counts_match':True,'seconds':time.monotonic()-started}
    write_json(out/'metrics.json',summary)
    if args.save_predictions:np.savez_compressed(out/'predictions.npz',names=np.asarray(names),masks=np.stack(masks_out))
    print(summary,flush=True)

if __name__=='__main__':main()
