"""Inspect a trained evidential model on a bundled digit or an 8x8 image."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from .evidential import MLP, variance_uncertainty


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('results/evidential_42.npz'))
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--index',type=int,default=None,help='Index in the bundled digits dataset')
    group.add_argument('--image',type=Path,help='An 8x8 grayscale digit, bright strokes on dark background')
    parser.add_argument('--invert',action='store_true',help='Invert a supplied image with dark strokes')
    args=parser.parse_args()
    model=MLP.load(args.model)
    if model.mode!='evidential':parser.error('the model must be evidential')
    label=None
    if args.image:
        with Image.open(args.image) as image:
            if image.size!=(8,8):parser.error('use an 8x8 image; automatic resizing could distort the digit')
            x=np.asarray(image.convert('L'),dtype=float).reshape(1,64)/255.
        if args.invert:x=1-x
    else:
        data=load_digits();index=args.index if args.index is not None else 0
        if not 0 <= index < len(data.target):parser.error('index out of range')
        x=data.data[index:index+1]/16.;label=int(data.target[index])
    u=variance_uncertainty(model.forward(x)[0])
    result=dict(predicted_digit=int(u['probabilities'][0].argmax()),true_digit=label,
                probabilities=u['probabilities'][0].tolist(),dirichlet_strength=float(u['strength'][0]),
                epistemic_uncertainty=float(u['epistemic'][0]),aleatoric_uncertainty=float(u['aleatoric'][0]),
                epistemic_by_class=u['epistemic_class'][0].tolist())
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
