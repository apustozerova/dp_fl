import numpy as np 
import pandas as pd

import algo
import federated
import scripts

import os
import json
import random

rand_seed = 42
#for al, l2 in [(0.001, 0.0001), (0.01, 0.0001), (0.001, 1e-5), (0.01, 1e-5)]:
#for l2 in [0.0001, 1e-5]:
for rand_seed in [42]:
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    x_target_train = np.load('data/rs'+str(rand_seed)+'_x_target_train.npy')
    y_target_train = np.load('data/rs'+str(rand_seed)+'_y_target_train.npy')
    x_target_test = np.load('data/rs'+str(rand_seed)+'_x_target_test.npy')
    y_target_test = np.load('data/rs'+str(rand_seed)+'_y_target_test.npy')
    n_classes = len(np.unique(y_target_train))

    for epsilon in [0.1, 1, 10, 100, 1000, 10000, 100000]:
    #for max_iter in [50]:

        number_of_clients = 2
        fl_iterations = 5
        data_per_client = int(x_target_train.shape[0]/number_of_clients)

        #create clients with set training parameters and datasets
        clients = {}
        for i in range(number_of_clients):
            clients[i] = algo.LogisticRegression_DPSGD()

            clients[i].n_classes      = n_classes
            clients[i].alpha          = 0.01
            clients[i].max_iter       = 100
            clients[i].lambda_        = 0.0001
            clients[i].tolerance      = 1e-5
            clients[i].sgdDP          = False
            clients[i].L              = 1 #should be 1 if DP == False
            clients[i].epsilon        = 1
            clients[i].C              = 1
            clients[i].outDP_local          = True
            clients[i].outDP_local_epsilon  = epsilon
#             clients[i].outDP_global         = False #not supported yet
#             clients[i].outDP_global_epsilon = 1 #not supported yet

            params = dict(clients[0].__dict__)

            clients[i].x = x_target_train[i*data_per_client:(i+1)*data_per_client]
            clients[i].y = y_target_train[i*data_per_client:(i+1)*data_per_client]

        fl_path = f'fl/rs{rand_seed}_ncl{number_of_clients}_fiter{fl_iterations}_lr{clients[0].alpha}_iter{clients[0].max_iter}_reg{clients[0].lambda_}'
        if clients[0].sgdDP:
            fl_path += f'_sgdDP{clients[0].sgdDP}_eps{clients[0].epsilon}_L{clients[0].L}_C{clients[0].C}'
        if clients[i].outDP_local:
            fl_path += f'_outDPlocal{clients[0].outDP_local}_eps{clients[0].outDP_local_epsilon}'
#         if clients[i].outDP_global:
#             fl_path += f'_outDPglobal{clients[0].outDP_global}_eps{clients[0].outDP_global_epsilon}'

        params.pop('x')
        params.pop('y')
        print(params)
        if os.path.exists(fl_path): 
            print('Experiment already exists:\n', fl_path)
        else:
            print('Creating new folder:\n', fl_path)
            os.mkdir(fl_path)
            with open(fl_path+'/params.json', 'w') as file:
                json.dump(params, file)
            results = {}
            for iteration in range(fl_iterations):

                print(iteration, ' FL iteration')
                for i in clients:
                    print("Start training client: ", i)
                    federated.train_client(iteration, clients[i], x_target_test, y_target_test)
                    if clients[i].outDP_local:
                        print('Adding local output DP')
                        federated.output_DP_federated(clients[i],  clients[i].x.shape[0], clients[i].outDP_local_epsilon)
                        clients[i].train_acc_outDP_local = clients[i].evaluate(clients[i].x, clients[i].y, acc=True)
                        clients[i].test_acc_outDP_local = clients[i].evaluate(x_target_test, y_target_test, acc=True)
                        np.save(fl_path + f'/i{iteration}_c{i}', clients[i].theta_before_noise)
                        np.save(fl_path + f'/i{iteration}_c{i}_outDP', clients[i].theta)
                        results[f'i{iteration}_c{i}'] = (clients[i].train_acc,  clients[i].test_acc, clients[i].train_acc_outDP_local, clients[i].test_acc_outDP_local)
                    else:
                        np.save(fl_path + f'/i{iteration}_c{i}', clients[i].theta)
                        results[f'i{iteration}_c{i}'] = (clients[i].train_acc,  clients[i].test_acc)
                        
                global_model = federated.aggregate(clients)
    #             if clients[0].outDP_global:
    #                 clients[0].theta = global_model  
    #                 clients[0] = output_DP_federated(clients[0],  clients[0].x.shape[0], clients[i].outDP_global_epsilon)
                np.save(fl_path + f'/i{iteration}_g', global_model)
                federated.update_clients(clients, global_model)
                #global model evaluation
                print('Global model evaluataion:')
                gtrain_acc = clients[0].evaluate(x_target_train, y_target_train, acc=True)
                gtest_acc = clients[0].evaluate(x_target_test, y_target_test, acc=True)
                results[f'i{iteration}_g'] = (gtrain_acc,  gtest_acc)
                if False and clients[0].evaluate(x_target_test, y_target_test)>=0.56:
                    break
            
            if clients[i].outDP_local:
                res = pd.DataFrame.from_dict(results, orient='index', columns=['train_acc', 'test_acc', 'train_acc_outDP', 'test_acc_out_DP'])
            else:    
                res = pd.DataFrame.from_dict(results, orient='index', columns=['train_acc', 'test_acc'])                
            res.to_csv(fl_path + f'/results.csv')
