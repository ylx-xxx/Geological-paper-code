"""Build content manifests and detect exact cross-split image duplication."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .core import pairs, read_h5, sha256, write_json

def cross_split_duplicates(rows):
    by_hash=defaultdict(list)
    for r in rows:by_hash[r['array_sha256']].append({'split':r['split'],'sample':r['sample']})
    return [items for items in by_hash.values() if len({x['split'] for x in items})>1]

def main():
    import hashlib
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];stats={}
    for split in ['train','val','test']:
        positive=valid=empty=0
        for ip,mp in pairs(args.data_root,split):
            x=read_h5(ip,'img');y=np.squeeze(read_h5(mp,'mask'))
            if not np.isin(y,[0,1,255]).all():raise ValueError(f'Invalid labels: {mp}')
            fg=int((y==1).sum());n=int((y!=255).sum());positive+=fg;valid+=n;empty+=fg==0
            h=hashlib.sha256();h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
            rows.append({'split':split,'sample':ip.name,'image_sha256':sha256(ip),'mask_sha256':sha256(mp),
                         'array_sha256':h.hexdigest(),'positive_pixels':fg,'valid_pixels':n})
        stats[split]={'samples':sum(r['split']==split for r in rows),'positive_pixels':positive,
                      'valid_pixels':valid,'positive_fraction':positive/valid,'empty_patches':empty}
        print(split,stats[split],flush=True)
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    duplicates=cross_split_duplicates(rows)
    write_json(out/'audit.json',{'splits':stats,'exact_cross_split_duplicates':duplicates,
        'spatial_independence':'NOT established by content hashes; georeferencing/event metadata required'})

if __name__=='__main__':main()
