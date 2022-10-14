
from densedrop import densedrop
import tensorflow as tf
from tensorflow import keras

import numpy as np

from loaddata import loaddata

from sklearn.metrics import roc_auc_score

import os

alpha=0.5
lr=0.01


x,tx,ty=loaddata()

#for optdigits_clean data already normalised
mn,mx=np.min(x,axis=0),np.max(x,axis=0)
x=(x-mn)/(mx-mn)
tx=(tx-mn)/(mx-mn)

def trainone(dex,x,tx,ty):
    
    pth=f'results/{dex}'
    os.makedirs(pth,exist_ok=True)

    dim0=int(x.shape[1])
    diml=int(np.ceil(np.sqrt(dim0)))
    dims=[alpha,alpha**2,alpha**3,alpha**2,alpha,1]
    #dims=[int((dim0**zw)*(diml**(1-zw))) for zw in dims]
    dims=[max([3,int(zw*dim0)]) for zw in dims]
    #print(dims)
    #exit()
    
    def genlayer(dim,act="relu"):
        return densedrop(dim,activation=act),keras.layers.Dense(dim,activation=act)
    def genlayers(dims):
        layers=[]
        for dim in dims[:-1]:
            layers.append(genlayer(dim))
        layers.append(genlayer(dims[-1],act="linear"))
        return layers
    def genmodel(inp,inp2,layers):
        q=inp
        q2=inp2
        for layer in layers:
            q=layer[0](q)
            q2=layer[1](q2)
        return keras.models.Model(inp,q),keras.models.Model(inp2,q2)
    
    inp=keras.layers.Input(shape=x.shape[1:])
    inp2=keras.layers.Input(shape=x.shape[1:])
    layers=genlayers(dims)
    model,model2=genmodel(inp,inp2,layers)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss=keras.losses.mean_squared_error)
    model.summary()
    model2.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss=keras.losses.mean_squared_error)
    model2.summary()
    #exit()
    #model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    #              loss=keras.losses.mean_squared_error)
    #model2.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    #              loss=keras.losses.mean_squared_error)
    
    model.fit(x,x,
                epochs=300,
                batch_size=100,
                validation_split=0.1,
                verbose=1,
                shuffle=True)#,
                #callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)])

    px= model.predict(x)
    dx=(px-x)**2
    dx=np.mean(dx,axis=1)
    div=np.std(dx)
    
    wei=model.get_weights()
    alts=[lay[0].get_matrix() for lay in layers]
    for i,w in enumerate(wei):
        if str(alts[0].shape)==str(w.shape):
            wei[i]=alts.pop(0)
            if len(alts)==0:
                break
    
    model2.set_weights(wei)
    
    p=model2.predict(tx)
    d=(p-tx)**2
    d=np.mean(d,axis=1)
    
    
    auc=roc_auc_score(ty,d)
    print(auc)

    model2.save(f"{pth}/model.h5")
    model2.save(f"{pth}/saved")
    os.system(f"python3 -m tf2onnx.convert --saved-model {pth}/saved --output {pth}/conv.onnx")
    np.savez_compressed(f"{pth}/result.npz",y_true=ty,y_score=d,div=div)
    with open(f"{pth}/auc","w") as f:
        f.write(str(auc))
    return model2,d,ty,auc
    
def train_many(fro,too,x,tx,ty):
    for i in range(fro,too):
        for j in range(5):print("-----------------------")
        print("training",i,(i-fro)/(too-fro))
        for j in range(5):print("-----------------------")
        trainone(i,x,tx,ty)


if __name__=="__main__":
    #trainone(0,x,tx,ty)
    #train_many(0,100,x,tx,ty)
    train_many(100,2000,x,tx,ty)
    #trainone(1000,x,tx,ty)





    
