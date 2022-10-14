import numpy as np
import json

from loaddata import loaddata
(x,_),(_,_)=loaddata()

num_feat=int(x.shape[1])

with open("hyper.json","r") as f:
    hyper=json.loads(f.read())




bag=hyper["bag"]

def features_of_index(i=0):
    np.random.seed(i)
    seed=np.random.randint(10000000)

    np.random.seed(seed)

    base=list(range(num_feat))

    np.random.shuffle(base)

    return base[:bag]




if __name__=="__main__":
    import sys
    i=0
    if len(sys.argv)>1:i=int(sys.argv[1])
    feat=features_of_index(i)


    print(feat)


