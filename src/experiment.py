"""Reproducible clean/corrupted digit experiment, with no dataset download."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .evidential import MLP, variance_uncertainty, entropy_uncertainty


def split_indices(labels, seed):
    trainval, test = train_test_split(np.arange(len(labels)), test_size=.2,
                                     stratify=labels, random_state=seed)
    train, val = train_test_split(trainval, test_size=.25,
                                 stratify=labels[trainval], random_state=seed)
    return train, val, test


def classification_metrics(p, y):
    pred, confidence = p.argmax(1), p.max(1)
    correct = pred == y
    edges = np.linspace(0,1,16)
    ece = 0.
    for i in range(15):
        mask = (confidence >= edges[i]) & ((confidence <= edges[i+1]) if i == 14 else (confidence < edges[i+1]))
        if mask.any(): ece += mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return dict(accuracy=float(correct.mean()), macro_f1=float(f1_score(y,pred,average='macro')),
                nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-15)).mean()),
                brier=float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean()),
                ece=float(ece))


def error_auc(score, y, p):
    errors = (p.argmax(1) != y).astype(int)
    return float(roc_auc_score(errors, score)) if len(np.unique(errors))==2 else None


def run(output=Path('results'), seeds=(42,43,44), epochs=250):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data = load_digits()
    x, y = data.data.astype(float)/16., data.target
    rows, all_histories = [], {}
    with threadpool_limits(limits=1):
        for seed in seeds:
            train, val, test = split_indices(y, seed)
            np.savez_compressed(output/f'split_{seed}.npz', train=train, validation=val, test=test)
            rng = np.random.default_rng(seed+2000)
            # Corruption is added after splitting, only to the held-out images.
            noise = rng.normal(size=x[test].shape)
            conditions = {'clean':x[test], 'noise_0.3':np.clip(x[test]+.3*noise,0,1),
                          'noise_0.6':np.clip(x[test]+.6*noise,0,1)}
            for mode in ('softmax','evidential'):
                model = MLP(mode=mode,seed=seed)
                history = model.fit(x[train],y[train],x[val],y[val],epochs=epochs)
                all_histories[f'{mode}_{seed}'] = history
                model.save(output/f'{mode}_{seed}.npz')
                for condition, xt in conditions.items():
                    out = model.forward(xt)[0]
                    p = model.probabilities(xt)
                    row = dict(seed=seed, method=mode, condition=condition,
                               n_test=len(test), best_epoch=model.best_epoch, **classification_metrics(p,y[test]))
                    row['error_auc_confidence'] = error_auc(1-p.max(1),y[test],p)
                    prediction_data = dict(indices=test, labels=y[test], probabilities=p)
                    if mode == 'evidential':
                        u, e = variance_uncertainty(out), entropy_uncertainty(out)
                        row.update(mean_strength=float(u['strength'].mean()),
                                   mean_aleatoric=float(u['aleatoric'].mean()),
                                   mean_epistemic=float(u['epistemic'].mean()),
                                   error_auc_epistemic=error_auc(u['epistemic'],y[test],p),
                                   error_auc_entropy_epistemic=error_auc(e['epistemic'],y[test],p))
                        prediction_data.update(alpha=out, epistemic=u['epistemic'],
                                               aleatoric=u['aleatoric'], epistemic_class=u['epistemic_class'])
                    np.savez_compressed(output/f'predictions_{mode}_{condition}_{seed}.npz',**prediction_data)
                    rows.append(row)
                    print(f'seed={seed} {mode:10s} {condition:10s} accuracy={row["accuracy"]:.4f} NLL={row["nll"]:.4f}',flush=True)
                if seed == seeds[0]:
                    if mode == 'evidential':
                        plot_diagnostics(model, x[test], y[test], conditions['noise_0.6'], output)
                    fig,ax=plt.subplots(figsize=(5.4,4.4))
                    cm=confusion_matrix(y[test],model.probabilities(x[test]).argmax(1))
                    im=ax.imshow(cm,cmap='Blues')
                    ax.set(xlabel='Predicted digit',ylabel='True digit',title=f'{mode.capitalize()} - clean test',xticks=range(10),yticks=range(10))
                    fig.colorbar(im,ax=ax)
                    fig.tight_layout();fig.savefig(output/f'confusion_{mode}.png',dpi=180);plt.close(fig)
    aggregates=[]
    for mode in ('softmax','evidential'):
        for condition in conditions:
            group=[r for r in rows if r['method']==mode and r['condition']==condition]
            agg=dict(method=mode,condition=condition)
            for key in group[0]:
                if key not in ('seed','method','condition','n_test','best_epoch') and all(r[key] is not None for r in group):
                    values=[r[key] for r in group]
                    agg[key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.)
            aggregates.append(agg)
    config=dict(seeds=list(seeds),epochs=epochs,hidden=128,learning_rate=.003,batch_size=64,
                regularization=.01,annealing_epochs=10,split='60/20/20 stratified',
                n_train=len(train),n_validation=len(val),n_test=len(test),
                dataset='sklearn load_digits (UCI optical handwritten digits)',
                dataset_sha256=hashlib.sha256(data.data.tobytes()+y.tobytes()).hexdigest(),
                noise_std=[0,.3,.6],selection='minimum validation predictive NLL',
                scope='classification and uncertainty core; no active domain adaptation',
                date='2026-10-06')
    payload=dict(config=config,runs=rows,aggregate=aggregates)
    (output/'metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (output/'history.json').write_text(json.dumps(all_histories,indent=2)+'\n')
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    plot_summary(payload,all_histories,output)
    return payload


def plot_summary(payload,histories,output):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    labels=['Clean','Noise 0.3','Noise 0.6']
    for mode,color in [('softmax','#607d8b'),('evidential','#167d9a')]:
        agg=[r for r in payload['aggregate'] if r['method']==mode]
        axes[0].errorbar(range(3),[r['accuracy']['mean']*100 for r in agg],
                         yerr=[r['accuracy']['std']*100 for r in agg],marker='o',capsize=4,label=mode,color=color)
        h=histories[f'{mode}_{payload["config"]["seeds"][0]}']
        axes[1].plot([v['epoch'] for v in h],[v['val_nll'] for v in h],label=mode,color=color)
    axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Test accuracy (%)',title='Mean and sample SD over seeds')
    axes[1].set(xlabel='Epoch',ylabel='Validation NLL',title='Checkpoint selection (first seed)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'summary.png',dpi=180);plt.close(fig)


def plot_diagnostics(model,clean,y,noisy,output):
    alpha=model.forward(clean)[0]
    u=variance_uncertainty(alpha)
    pred=u['probabilities'].argmax(1)
    errors=pred!=y
    # Show actual difficult images, including mistakes when available.
    order=np.argsort(u['epistemic'])[::-1][:8]
    fig,axes=plt.subplots(2,4,figsize=(9,4.5))
    for ax,i in zip(axes.flat,order):
        ax.imshow(clean[i].reshape(8,8),cmap='gray_r',vmin=0,vmax=1)
        ax.set_title(f'True {y[i]} / predicted {pred[i]}\nEU {u["epistemic"][i]:.4f}',fontsize=10)
        ax.axis('off')
    fig.suptitle('Eight clean test images with highest epistemic uncertainty')
    fig.tight_layout();fig.savefig(output/'uncertain_examples.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    bins=np.linspace(0,max(u['epistemic'].max(),1e-6),25)
    for mask,label in [(~errors,'Correct'),(errors,'Incorrect')]:
        if mask.any():axes[0].hist(u['epistemic'][mask],bins=bins,alpha=.65,label=f'{label} (n={mask.sum()})')
    axes[0].set(xlabel='Epistemic uncertainty (covariance trace)',ylabel='Count',title='Clean error separation');axes[0].legend()
    un=variance_uncertainty(model.forward(noisy)[0])
    axes[1].bar(['Clean AU','Noisy AU','Clean EU','Noisy EU'],
                [u['aleatoric'].mean(),un['aleatoric'].mean(),u['epistemic'].mean(),un['epistemic'].mean()],color=['#167d9a','#d59136']*2)
    axes[1].set(ylabel='Mean uncertainty',title='Noise std 0.6; AU and EU use same units')
    fig.tight_layout();fig.savefig(output/'uncertainty.png',dpi=180);plt.close(fig)
    i=int(order[0]);fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].bar(range(10),u['epistemic_class'][i]);axes[0].set(xticks=range(10),xlabel='Digit class',ylabel='Epistemic variance',title=f'Class uncertainty: true {y[i]}, predicted {pred[i]}')
    im=axes[1].imshow(u['correlation'][i],vmin=-1,vmax=1,cmap='coolwarm')
    axes[1].set(xticks=range(10),yticks=range(10),title='Between-class correlation');fig.colorbar(im,ax=axes[1])
    fig.tight_layout();fig.savefig(output/'class_uncertainty.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--epochs',type=int,default=250)
    args=parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):parser.error('positive epochs and unique seeds required')
    run(args.output,tuple(args.seeds),args.epochs)
