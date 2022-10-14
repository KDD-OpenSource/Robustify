import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from sklearn.metrics import roc_auc_score as auc

import numpy as np
import matplotlib.pyplot as plt
import json
import time

#load hyperparameters
with open("hyper.json","r") as f:
    hyper=json.loads(f.read())

#define the normal dataset
dataset=int(hyper["dataset"])

#enumerate each repetition
import sys
dex=0
if len(sys.argv)>1:
    dex=int(sys.argv[1])

#save each model in its own folder
pth=f"results/{dex}/"
import os
import shutil
if os.path.isdir(pth):
    shutil.rmtree(pth)
os.makedirs(pth, exist_ok=False)

#initialise each model differentiate to gain reproducability
seed=dex
tf.random.set_seed(seed)
np.random.seed(seed)

#load the dataset and reshape it
from loaddata import loaddata
(x_train, y_train), (x_test, y_test) = loaddata()
if len(x_train.shape)==3:x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
if len(x_test.shape)==3:x_test =np.reshape(x_test ,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

#implement the feature bagging
seed=np.random.randint(10000000)
print(seed)

np.random.seed(seed)
x_train=x_train.transpose()
np.random.shuffle(x_train)
x_train=x_train.transpose()
np.random.seed(seed)
x_test=x_test.transpose()
np.random.shuffle(x_test)
x_test=x_test.transpose()

dim=hyper["bag"]

x_train=x_train[:,:dim]
x_test=x_test[:,:dim]

mn=np.min(x_train,axis=0)

mx=np.max(x_train,axis=0)


def normalise(q):
  """removes the mean of each feature"""
  q=np.array(q)
  q=(q-mn)/(mx-mn)
  return q

def getdata(x,y,norm=True,normdex=7,n=-1):
  """Returns a subset of features. x: data, y: label, norm: Return normal data?, normdex: Which number is considered normal, n: maximum amount of features returned"""
  if norm:
    ids=np.where(y==normdex)
  else:
    ids=np.where(y!=normdex)
  qx=x[ids]
  if n>0:qx=qx[:n]
  qy=np.reshape(qx,(int(qx.shape[0]),dim))
  return normalise(qy),qx

#split mnist data into normal and not normal data
train,rawtrain=getdata(x_train,y_train,norm=True,normdex=dataset)
at,rawat=getdata(x_test,y_test,norm=False,normdex=dataset)
t,rawt=getdata(x_test,y_test,norm=True,normdex=dataset)



def getmodel(q,reg=None,act="relu",mean=1.0,seed=None):
  """defines a tensorflow model: q: shape of layers, reg: Regularisation, act: Activation function, mean: desired mean, seed: seed """
  np.random.seed(seed)
  tf.random.set_seed(seed)
  inn=Input(shape=(dim,))
  w=inn
  for aq in q[1:-1]:
    w=Dense(aq,activation=act,use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
  w=Dense(q[-1],activation="linear",use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
  m=Model(inn,w,name="DEAN")
  zero=K.ones_like(w)*mean
  loss=mse(w,zero)
  loss=K.mean(loss)
  m.add_loss(loss)
  m.compile(Adam(lr=hyper["lr"]))
  return m

#define the tensorflow model
l=[dim for i in range(hyper["depth"])]
l=[32,10,10,1]
m=getmodel(l,reg=None,act="relu",mean=1.0,seed=seed)

m.summary()


callb=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                   keras.callbacks.TerminateOnNaN(),
                   keras.callbacks.ModelCheckpoint(f"{pth}/model.tf", monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True),
                   keras.callbacks.CSVLogger(f"{pth}/history.csv")]


m.summary()



#train the model
h=m.fit(train,None,
        epochs=500,
        batch_size=hyper["batch"],
        validation_split=0.25,
        verbose=1,
        callbacks=callb)

m.save(f"{pth}/saved")

os.system(f"python3 -m tf2onnx.convert --saved-model {pth}/saved --output {pth}/model.onnx")



#predict the output of each sample 
model=h.model
p=model.predict(t)#test normal
w=model.predict(at)#test abnormal
pain=model.predict(train)#train

#as the network is multidimensional, we need to average it
pp=np.mean(p,axis=-1)
ppain=np.mean(pain,axis=-1)
ww=np.mean(w,axis=-1)

#extract the constant close from the prediction on the train dataset. This is a function of the underlying datadistribution and thus easiest measured
m=np.mean(ppain)

#score each sample using this mean
pd=np.abs(pp-m)
wd=np.abs(ww-m)

#calculate the auc score of a single model
y_score=np.concatenate((pd,wd))
y_true=np.concatenate((np.zeros_like(pp),np.ones_like(ww)))
auc_score=auc(y_true,y_score)
print(f"reached auc of {auc_score}")

with open(f"{pth}/auc","w") as f:
  f.write(str(auc_score)+"\n")


##save everything that could be needed
#np.savez_compressed(f"{pth}/result.npz",y_true=y_true,y_score=y_score,pd=pd,wd=wd,pdm=pdm,wdm=wdm,t=t,at=at,rawt=rawt,rawat=rawat,train=train,rawtrain=rawtrain)

#save the variables needed to combine multiple models
np.savez_compressed(f"{pth}/result.npz",y_true=y_true,y_score=y_score, m=m,pain=pain)









