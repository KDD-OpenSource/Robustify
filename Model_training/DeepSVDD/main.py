from loaddata import loaddata

#from pyod.models.deep_svdd import DeepSVDD
from rawsvdd import DeepSVDD

from sklearn.metrics import roc_auc_score

import numpy as np

import os
import json


key=None
import sys
if len(sys.argv)>1:
    key=sys.argv[1]

x,tx,ty=loaddata(key)

if key is None:
    key="use7"

mn,mx=np.min(x,axis=0),np.max(x,axis=0)

x=(x-mn)/(mx-mn)
tx=(tx-mn)/(mx-mn)


def run_one_svdd(dex,sdex,**kwargs):

    
    model = DeepSVDD(**kwargs)#gamma=0.1, alpha=1e-5, beta=1e-5, nu=0.1, batch_size=100, epochs=100, optimizer='adam', verbose=True)
    
    
    model.fit(x)
    
    p=model.decision_function(tx)
    
    auc=roc_auc_score(ty,p)
    
    fold='./models/'+str(dex)+'/'+str(sdex)
    os.makedirs(fold,exist_ok=True)
    model.model_.save(fold+'/saved')
    with open(fold+'/params.json','w') as f:
        json.dump(kwargs,f,indent=2)
    with open(fold+'/auc.json','w') as f:
        json.dump(auc,f)

    model.pmodel.save(fold+'/pmodel')

    with open(fold+"/c.json","w") as f:
        json.dump([float(zw) for zw in model.c],f)

    pred=model.model_.predict(x)
    border=float(np.quantile(pred,0.8))
    with open(fold+"/border.json","w") as f:
        json.dump(border,f)


    os.system(f"python3 -m tf2onnx.convert --saved-model {fold}/saved --output {fold}/merged.onnx")
    os.system(f"python3 -m tf2onnx.convert --saved-model {fold}/pmodel --output {fold}/pmodel.onnx")




    return auc,model

def run_n_svdd(dex,n=10,**kwargs):
    aucs=[run_one_svdd(dex,i,**kwargs)[0] for i in range(n)]
    return np.mean(aucs),np.std(aucs)
    


if __name__=="__main__":
    #auc,model=run_one_svdd(0,0,batch_size=32,verbose=True,l2_regularizer=0.0,dropout_rate=0.2,epochs=50,optimizer="adam",output_activation="linear",hidden_activation="relu",hidden_neurons=[30,15],preprocessing=False)
    auc,model=run_n_svdd(key,batch_size=32,verbose=True,l2_regularizer=0.0,dropout_rate=0.2,epochs=50,optimizer="adam",output_activation="linear",hidden_activation="relu",hidden_neurons=[30,15],preprocessing=False)
    print(auc)

    exit()

    m=model.model_
    l=m.layers
    w=m.weights
    

    c=model.c
    p=model.pmodel

    true=m.predict(tx)
    pred=p.predict(tx)
    pred=np.sum((pred-c)**2,axis=-1)

    
                                                                                                                                                                                                                                                                                                                                            
