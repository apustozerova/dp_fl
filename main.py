import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import algo
# import attack

import torch
import os
import json

rand_seed=42
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
   
raw_data_path = '../datasets/dataset_purchase'
raw_data = pd.read_csv(raw_data_path)
y=raw_data['63']
X_raw =raw_data.drop('63', axis=1)
y_raw =  y.replace(100, 0)
print('Dataset: ', raw_data_path)
print('Classes in classification task: ', y.nunique())
n_classes = y.nunique()

# X_train, x_shadow, y_train, y_shadow = train_test_split(X, y, train_size=0.2, random_state=rand_seed)
# print(X_train.shape, x_shadow.shape)

for rand_seed in [42]: #1,3,13,24,42]:

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    x_target_train = np.load('data/rs'+str(rand_seed)+'_x_target_train.npy')
    y_target_train = np.load('data/rs'+str(rand_seed)+'_y_target_train.npy')
    x_target_test = np.load('data/rs'+str(rand_seed)+'_x_target_test.npy')
    y_target_test = np.load('data/rs'+str(rand_seed)+'_y_target_test.npy')
    # x_shadow = np.load('data/rs'+str(rand_seed)+'_x_shadow')
    # y_shadow = np.load('data/rs'+str(rand_seed)+'_y_shadow')

    #for epsilon in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,30,50,70,100]:
    #for lam in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
    for iter in [700,800,900,1000]: 
    #for C in [0.5,1,1.3,1.5,1.7,2,2.5,3,4,5]:
               
        model = algo.LogisticRegression_DPSGD()

        model.n_classes      = n_classes
        model.alpha          = 0.01
        model.max_iter       = iter
        model.lambda_        = 0.0001
        model.tolerance      = 1e-5
        model.sgdDP          = True
        model.L              = 50
        model.epsilon        = 1000000
        model.C              = 2

        params = dict(model.__dict__) #save model's parameters to json file later

        tm_path = f'tm/rs{rand_seed}_lr{model.alpha}_iter{model.max_iter}_reg{model.lambda_}_DP{model.sgdDP}'
        if model.sgdDP:
            tm_path += f'_eps{model.epsilon}_L{model.L}_C{model.C}'
        
        if not os.path.exists(tm_path+'_target_model_params.json'):
            print("Start training")
            X,y = model.init_theta(x_target_train, y_target_train)
            model.train(X,y)
            params['train_acc'] = model.evaluate(x_target_train, y_target_train, acc=True)
            params['test_acc'] = model.evaluate(x_target_test, y_target_test, acc=True)
            
            np.save(tm_path+'_target_model', model.theta)
            with open(tm_path+'_target_model_params.json', 'w') as file:
                json.dump(params, file)

        else:
            print('Model already exists')
        print('Model path: ',tm_path)
           
