"""Verify saved TRAIN heads without fitting, tuning or loading test labels."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
models=json.loads((R/'data/frozen_training_models.json').read_text())
with np.load(R/'data/train_features.npz',allow_pickle=False) as z:
    data={k:z[k] for k in z.files}
assert len(data['sample'])==384 and len(set(data['image']))==192
def design(head, x):
    z = (np.asarray(x, float) - head['mean']) / head['scale']
    z /= np.maximum(1., np.linalg.norm(z, axis=1, keepdims=True) / 5.)
    return z if head['odd'] else np.c_[np.ones(len(z)), z]

rows = []
for rep, block in models.items():
    for e, expert in enumerate(block['experts']):
        assert expert['active']
        a_images, b_images = set(expert['A_images']), set(expert['B_images'])
        assert len(a_images) == len(b_images) == 96 and not a_images & b_images
        pair_images = data['image'][data['pair_query']]
        aa = np.flatnonzero(np.isin(pair_images, list(a_images)))
        bb = np.flatnonzero(np.isin(pair_images, list(b_images)))
        support = [a for a in range(2, 9)
                   if (data['pair_action'][aa] == a).sum() >= 2
                   and (data['pair_action'][bb] == a).sum() >= 2]
        assert support == expert['support']
        aa = aa[np.isin(data['pair_action'][aa], support)]
        bb = bb[np.isin(data['pair_action'][bb], support)]
        jj = aa[data['valid'][data['pair_query'][aa]]]
        xb = data['pair_' + rep][bb]
        accepted = ((design(expert['u'], xb) @ expert['u']['theta'] > 0)
                    & (design(expert['j'], xb) @ expert['j']['theta'] >= 0))
        target = accepted * data['pair_du'][bb]
        assert int((target < 0).sum()) == expert['negative_delivered_targets']
        for name, ids, labels, feature in [
            ('u', aa, data['pair_du'][aa], 'pair_' + rep),
            ('j', jj, data['pair_dj'][jj], 'pair_' + rep),
            ('mean', bb, data['pair_du'][bb], 'pair_pre'),
            ('delivery', bb, target, 'pair_pre')]:
            h = expert[name]
            x = np.asarray(data[feature][ids], float)
            odd = name in ('u', 'j')
            assert h['odd'] == odd
            mean = np.zeros(x.shape[1]) if odd else x.mean(0)
            scale = np.maximum(np.sqrt((x*x).mean(0)) if odd else x.std(0),
                               1e-4 if odd else .05)
            np.testing.assert_allclose(mean, h['mean'], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(scale, h['scale'], rtol=1e-12, atol=1e-14)
            z = design(h, x)
            theta = np.asarray(h['theta'])
            assert z.shape[1] == (39 if odd else 89)
            if odd:
                np.testing.assert_array_equal(design(h, -x), -z)
            grad = z.T @ (z @ theta - labels) / len(z) + .01 * theta
            gap = float(grad @ grad / .02)
            assert gap <= 1e-12 and len(z) == h['proof']['fit_rows']
            rows.append(dict(representation=rep, expert=e, head=name,
                             fit_rows=len(z), design_dimension=z.shape[1],
                             gradient_norm=float(np.linalg.norm(grad)),
                             objective_gap_upper=gap, odd=odd))

print(json.dumps(dict(status="PASS",saved_heads=len(rows),new_fits=0,max_gap=max(r["objective_gap_upper"] for r in rows))))
