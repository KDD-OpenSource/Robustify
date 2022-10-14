import numpy as np

import os

print(os.getcwd())

def loaddata():
    c=os.getcwd()
    c=c[c.rfind("/"):]
    f=np.load(f"/home/simon/DATA/{c}.npz")
    x,tx,ty=f["x"],f["tx"],f["ty"]
    return x,tx,ty


if __name__ == "__main__":
    x,tx,ty=loaddata()
    print(x.shape,tx.shape,ty.shape)
    print(np.mean(ty))
    print(np.unique(ty))


