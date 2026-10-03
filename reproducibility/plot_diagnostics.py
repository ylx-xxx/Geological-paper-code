"""Plot diagnostics from one audited evaluation; never run a separate inference path."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import write_json

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();src=Path(args.evaluation);out=Path(args.out)
    frame=pd.read_csv(src/'per_sample.csv');summary=json.loads((src/'metrics.json').read_text())
    for key in ['TN','FP','FN','TP']:
        if int(frame[key].sum())!=summary[key]:raise ValueError(f'Global/sample {key} mismatch')
    out.mkdir(parents=True,exist_ok=False)
    cm=np.array(summary['confusion_matrix'],dtype=float)
    normalized=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)!=0)
    fig,ax=plt.subplots(figsize=(4,3.6));ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{normalized[i,j]:.2%}',ha='center',va='center',color='white' if normalized[i,j]>.5 else '#123652')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Background','Landslide'],yticklabels=['Background','Landslide'],xlabel='Prediction',ylabel='Ground truth')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.pdf');fig.savefig(out/'confusion_matrix.png',dpi=300);plt.close(fig)
    positive=frame.loc[frame.positive_pixels>0].copy()
    positive['coverage']=100*positive.positive_pixels/positive.valid_pixels
    positive['group']=pd.cut(positive.coverage,[0,1,3,5,np.inf],right=False,labels=['(0,1%)','[1,3%)','[3,5%)','[5,100%]'])
    groups=positive.groupby('group',observed=False).agg(IoU=('Landslide_IoU','mean'),F1=('Landslide_F1','mean'),Samples=('sample','count'))
    groups.to_csv(out/'coverage_groups.csv')
    fig,ax=plt.subplots(figsize=(5,3.5));groups[['IoU','F1']].plot.bar(ax=ax,color=['#2166ac','#7fafd2'],rot=0)
    ax.set(xlabel='Annotated landslide fraction per patch',ylabel='Mean patch score',ylim=(0,1));fig.tight_layout()
    fig.savefig(out/'coverage_groups.pdf');fig.savefig(out/'coverage_groups.png',dpi=300);plt.close(fig)
    empty=frame.loc[frame.positive_pixels==0]
    write_json(out/'provenance.json',{'source_metrics':summary,'positive_patches':len(positive),
        'empty_patches':len(empty),'empty_patch_false_positive_pixels':int(empty.FP.sum()),
        'empty_patches_with_false_positives':int((empty.FP>0).sum()),
        'note':'Patch coverage is not object size; no separate inference was performed.'})

if __name__=='__main__':main()
