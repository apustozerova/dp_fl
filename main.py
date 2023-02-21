import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import algo
import scripts
# import attack

import torch
import os
import json

rand_seed=42
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

dataset = 'loan'
# x_target_train, y_target_train, x_target_test, y_target_test = scripts.load_purchase(rand_seed)
x_target_train, y_target_train, x_target_test, y_target_test = scripts.load_loan(rand_seed, tr_size=10000)
    
# for rand_seed in [42]: #1,3,13,24,42]:
for lam in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    #for epsilon in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,30,50,70,100]:
    # for lam in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
    for iter in [10, 50, 100, 200, 300]: 
    #for C in [0.5,1,1.3,1.5,1.7,2,2.5,3,4,5]:
               
        model = algo.LogisticRegression_DPSGD()

        model.n_classes      = len(np.unique(y_target_test))
        model.alpha          = 0.01
        model.max_iter       = iter
        model.lambda_        = lam
        model.tolerance      = 1e-5
        model.sgdDP          = False
        model.L              = 1 #should be 1 if DP == False
        model.epsilon        = 1
        model.C              = 1
        model.outDP_local          = False
        model.outDP_local_epsilon  = 1

        params = dict(model.__dict__) #save model's parameters to json file later

        tm_path = f'{dataset}/centr/rs{rand_seed}_lr{model.alpha}_iter{model.max_iter}_reg{model.lambda_}'
        if model.sgdDP:
            tm_path += f'_sgdDP{model.sgdDP}_eps{model.epsilon}_L{model.L}_C{model.C}'
        if model.outDP_local:
            tm_path += f'_outDPlocal{model.outDP_local}_eps{model.outDP_local_epsilon}'
        
        if not os.path.exists(tm_path+'_target_model_params.json'):
            print("Start training")
            X,y = model.init_theta(x_target_train, y_target_train)
            model.train(X,y)
            params['train_acc'] = model.evaluate(x_target_train, y_target_train, acc=True)
            params['test_acc'] = model.evaluate(x_target_test, y_target_test, acc=True)
             
            if model.outDP_local:
                noise_model = scripts.output_DP(model,  x_target_train.shape[0], model.outDP_local_epsilon)
                params['train_acc_output_dp'] = noise_model.evaluate(x_target_train, y_target_train, acc=True)
                params['test_acc_output_dp'] = noise_model.evaluate(x_target_test, y_target_test, acc=True)
                
            np.save(tm_path+'_target_model', model.theta)
            with open(tm_path+'_target_model_params.json', 'w') as file:
                json.dump(params, file)

        else:
            print('Model already exists')
        print('Model path: ',tm_path)
           
