#merge with same number of events for each number

import os
import numpy as np

import sys

from simplestat import statinf
import json
from sklearn.metrics import roc_auc_score as auc

from uqdm import uqdm

#combine only cmax models
cmax=100000
if len(sys.argv)>1:
    cmax=int(sys.argv[1])

if not os.path.isdir("results"):exit()

fns=["results/"+zw+"/result.npz" for zw in os.listdir("results")[:cmax+10]]#there is a limit of 1000 files open at the same time
fns=[zw for zw in fns if os.path.isfile(zw)]

y_true=None
y_scores=[]

for fn,func in uqdm(fns):
    func(fn)
    f=np.load(fn,allow_pickle=True)
    if y_true is None:
        y_true=f["y_true"]
    y_scores.append(f["y_score"])

y_scores=y_scores[:cmax]

y_scores=np.array(y_scores)

aucs=[auc(y_true,y_score) for y_score in y_scores]
print("single models:")
print(json.dumps(statinf(aucs),indent=2))


y_score=np.sqrt(np.mean(y_scores**2,axis=0))


auc_score=auc(y_true,y_score)

print("ensemble:",auc_score)

#try:
for i in range(1):
    c=os.getcwd()
    c=c[c.rfind("/"):]
    with open(f"../stats/{c}","w") as f:
        f.write(str(auc_score))
#except:
#    print("cant write stats")



