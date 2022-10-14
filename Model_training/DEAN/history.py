import numpy as np
import os

from sklearn.metrics import roc_auc_score
from simplestat import statinf

import json

from plt import *

fns=[f"results/{zw}/result.npz" for zw in sorted([int(zw) for zw in os.listdir("results")])]
fns=[fn for fn in fns if os.path.isfile(fn)]

fs=[np.load(fn) for fn in fns]

y_true=fs[0]["y_true"]
y_scores=np.array([f["y_score"] for f in fs])


def addmedian(q):
    while len(q.shape)>1:
        q=np.median(q,axis=0)
    return q

y_scores=np.abs(y_scores)**2
y_scores=np.cumsum(y_scores,axis=0)
#y_scores=np.array([addmedian(y_scores[:i]) for i in range(1,1+y_scores.shape[0])])

#y_scores=np.cumsum(np.abs(y_scores),axis=0)

aucs=[roc_auc_score(y_true,y_score) for y_score in y_scores]

plt.plot(aucs)

#plt.xscale("log")

plt.show()


