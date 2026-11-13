"""Reproduce the original 16 TRAIN-only heads, never overwriting a frozen model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from pact import fitting
from pact.frozen_math import predict

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output; frozen records must not be overwritten')
    saved=json.loads((R/'data/frozen_training_models.json').read_text())
    with np.load(R/'data/train_features.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    result={}
    pair_images=data['image'][data['pair_query']]
    for rep,block in saved.items():
        experts=[]
        for original in block['experts']:
            A,B=set(original['A_images']),set(original['B_images'])
            assert len(A)==len(B)==96 and not A&B
            aa=np.flatnonzero(np.isin(pair_images,list(A)))
            bb=np.flatnonzero(np.isin(pair_images,list(B)))
            support=[a for a in range(2,9) if (data['pair_action'][aa]==a).sum()>=2 and (data['pair_action'][bb]==a).sum()>=2]
            assert support==original['support']
            aa=aa[np.isin(data['pair_action'][aa],support)]
            bb=bb[np.isin(data['pair_action'][bb],support)]
            jj=aa[data['valid'][data['pair_query'][aa]]]
            u=fitting.fit(data['pair_'+rep][aa],data['pair_du'][aa],True)
            j=fitting.fit(data['pair_'+rep][jj],data['pair_dj'][jj],True)
            x=data['pair_'+rep][bb]
            accepted=(predict(u,x)>0)&(predict(j,x)>=0)
            target=accepted*data['pair_du'][bb]
            assert int((target<0).sum())==original['negative_delivered_targets']
            mean=fitting.fit(data['pair_pre'][bb],data['pair_du'][bb],False)
            delivery=fitting.fit(data['pair_pre'][bb],target,False)
            expert=dict(original,u=u,j=j,mean=mean,delivery=delivery)
            for name in ['u','j','mean','delivery']:
                np.testing.assert_allclose(expert[name]['theta'],original[name]['theta'],rtol=1e-8,atol=1e-10)
            experts.append(expert)
        result[rep]=dict(block,experts=experts)
    assert fitting.FITS==16
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='MATCHES_FROZEN_TRAIN_HEADS',fits=16,test_data_read=False)))

if __name__=='__main__':main()
