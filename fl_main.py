import numpy as np
import pandas as pd

import algo
import federated

import os
import json

for rand_seed in [42]: #1,3,13,24,42:

    np.random.seed(rand_seed)
    
    x_target_train = np.load('data/rs'+str(rand_seed)+'_x_target_train.npy')
    y_target_train = np.load('data/rs'+str(rand_seed)+'_y_target_train.npy')
    x_target_test = np.load('data/rs'+str(rand_seed)+'_x_target_test.npy')
    y_target_test = np.load('data/rs'+str(rand_seed)+'_y_target_test.npy')
    n_classes = len(np.unique(y_target_train))

    for m_it in [1, 10, 50, 100, 150]:
    
        number_of_clients = 2
        fl_iterations = 5
        data_per_client = int(x_target_train.shape[0]/number_of_clients)

        #create clients with set training parameters and datasets
        clients = {}
        for i in range(number_of_clients):
            clients[i] = algo.LogisticRegression_DPSGD()

            clients[i].n_classes      = n_classes
            clients[i].alpha          = 0.01
            clients[i].max_iter       = m_it
            clients[i].lambda_        = 0.0001
            clients[i].tolerance      = 1e-5
            clients[i].DP             = True
            clients[i].L              = 1 #should be 1 if DP == False
            clients[i].epsilon        = 100000
            clients[i].C              = 1.25

            params = dict(clients[0].__dict__)

            clients[i].x = x_target_train[i*data_per_client:(i+1)*data_per_client]
            clients[i].y = y_target_train[i*data_per_client:(i+1)*data_per_client]

        fl_path = f'fl/rs{rand_seed}_ncl{number_of_clients}_fiter{fl_iterations}_lr{clients[0].alpha}_iter{clients[0].max_iter}_reg{clients[0].lambda_}_DP{clients[0].DP}'
        if clients[0].DP:
            fl_path += f'_eps{clients[0].epsilon}_L{clients[0].L}_C{clients[0].C}'


        params.pop('x')
        params.pop('y')
        print(params)
        if os.path.exists(fl_path):

            print('Experiment already exists')

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
                    federated.train_client(clients[i], x_target_test, y_target_test)
                    np.save(fl_path + f'/i{iteration}_c{i}', clients[i].theta)
                    results[f'i{iteration}_c{i}'] = (clients[i].train_acc,  clients[i].test_acc)


                global_model = federated.aggregate(clients)
                np.save(fl_path + f'/i{iteration}_g', global_model)
                federated.update_clients(clients, global_model)
                #global model evaluation
                print('Global model evaluataion:')
                gtrain_acc = clients[0].evaluate(x_target_train, y_target_train, acc=True)
                gtest_acc = clients[0].evaluate(x_target_test, y_target_test, acc=True)
                results[f'i{iteration}_g'] = (gtrain_acc,  gtest_acc)
                if clients[0].evaluate(x_target_test, y_target_test, acc=True)>=0.55:
                    break

            res = pd.DataFrame.from_dict(results, orient='index')
            res.to_csv(fl_path + f'/results.csv')
