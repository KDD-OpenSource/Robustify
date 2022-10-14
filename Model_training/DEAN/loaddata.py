import numpy as np
import json

import os


def loaddata():
    #f=np.load("optdigits_clean.npz")
    c=os.getcwd()
    c=c[c.rfind("/"):]
    f=np.load(f"/work/msimklue/DATA/{c}.npz")
    #f=np.load("satimage-2.npz")
    x,tx,ty=f["x"],f["tx"],f["ty"]
    y=np.zeros(len(x))
    return (x,y),(tx,ty)

if __name__=="__main__":
    (x,y),(tx,ty)=loaddata()

    print(x.shape)
