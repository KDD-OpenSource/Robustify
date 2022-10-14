import numpy as np


def loaddata(what=None):
    if what is None:
        f=np.load("satellite.npz")
    else:
        f=np.load(f"/home/simon/DATA/{what}.npz")
    x,tx,ty=f["x"],f["tx"],f["ty"]
    return x,tx,ty


if __name__ == "__main__":
    x,tx,ty=loaddata()
    print(x.shape,tx.shape,ty.shape)
    print(np.mean(ty))
    print(np.unique(ty))


