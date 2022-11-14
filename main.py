import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import algo
# import attack

# from torch import nn,optim
import torch

rand_seed=42
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
        

raw_data_path = '../datasets/dataset_purchase'
raw_data = pd.read_csv(raw_data_path)
y=raw_data['63']
X=raw_data.drop('63', axis=1)
y =  y.replace(100, 0)
print('Dataset: ', raw_data_path)
print('Classes in classification task: ', y.nunique())
n_classes = y.nunique()

X_train, x_shadow, y_train, y_shadow = train_test_split(X, y, train_size=0.2, random_state=rand_seed)
print(X_train.shape, x_shadow.shape)

#Target model
X_train_size = 10000
X_test_size = 10000
x_target_train = np.array(X_train[:X_train_size])
y_target_train = np.array(y_train[:X_train_size])
x_target_test = np.array(X_train[X_train_size:X_train_size+X_test_size])
y_target_test = np.array(y_train[X_train_size:X_train_size+X_test_size])
if y_target_test.shape[0]<X_test_size or y_target_train.shape[0]<X_train_size:
    raise ValueError(
            "Not enough traning or test data for the target model")

for L in [1,10,100]:
    for epsilon in np.arange(0,1,0.1):
        
        model = algo.LogisticRegression_DPSGD()

        model.n_classes      = n_classes
        model.alpha          = 0.001
        model.max_iter       = 100*X_train_size
        model.lambda_        = 1e-3
        model.tolerance      = 10e-5
        model.DP             = True
        model.L              = L
        model.epsilon        = round(epsilon,2)


        X,y = model.init_theta(x_target_train, y_target_train)
        model.train(X,y)
        model.evaluate(x_target_train, y_target_train, acc=True)
        model.evaluate(x_target_test, y_target_test, acc=True)

        tm_path = f'tm/lr{model.alpha}_iter{int(model.max_iter/X_train_size)}_reg{model.lambda_}_DP{model.DP}'
        if model.DP:
            tm_path += f'_eps{model.epsilon}_L{model.L}'
        np.save(tm_path+'_target_model', model.theta)

        print(tm_path)
#Shadow models
# s_ms = {}
# number_of_sms = 10
# shadow_size = 50000
# shadow_batch_size = int(shadow_size/number_of_sms)

# x_shadow_train = np.array(x_shadow[:shadow_size])
# y_shadow_train = np.array(y_shadow[:shadow_size])
# x_shadow_test = np.array(x_shadow[shadow_size:2*shadow_size])
# y_shadow_test = np.array(y_shadow[shadow_size:2*shadow_size])

# attack.train_shadow_models(number_of_sms,)

# for i in range(number_of_sms):  
#     batch_start = i*shadow_batch_size
#     batch_end = (i+1)*shadow_batch_size
    
#     shadow_model = algo.LogisticRegression_DPSGD()

#     shadow_model.n_classes      = n_classes
#     shadow_model.alpha          = 0.001
#     shadow_model.max_iter       = 100*shadow_batch_size
#     shadow_model.lambda_        = 10e-3
#     shadow_model.tolerance      = 10e-5
#     shadow_model.DP             = False

#     X,y = shadow_model.init_theta(x_shadow_train[batch_start:batch_end], y_shadow_train[batch_start:batch_end] )
#     shadow_model.SGD(X,y)
#     print('Shadow model: ', i)
#     shadow_model.evaluate(x_shadow_train[batch_start:batch_end], y_shadow_train[batch_start:batch_end])
#     shadow_model.evaluate(x_shadow_test[batch_start:batch_end], y_shadow_test[batch_start:batch_end])
#     s_ms[i] = shadow_model

# #Attack model

# shadow_train_pred = []
# shadow_test_pred = []

# for i in range(number_of_sms): 
#     batch_start = i*shadow_batch_size
#     batch_end = (i+1)*shadow_batch_size
    
#     train_prediciton = s_ms[i].predict(x_shadow_train[batch_start:batch_end], y_shadow_train[batch_start:batch_end])
#     test_prediciton = s_ms[i].predict(x_shadow_test[batch_start:batch_end], y_shadow_test[batch_start:batch_end])
    
#     shadow_train_pred.append(train_prediciton)
#     shadow_test_pred.append(test_prediciton)
    
    