import numpy as np
from sklearn.preprocessing import OneHotEncoder


def aggregate(clients):
    
    global_model = clients[0].theta 
    for i in range(1, len(clients.keys())):
        global_model += clients[i].theta 

    return global_model/len(clients.keys())

def update_clients(clients, global_model):
    
    for i in clients:
        clients[i].theta = global_model


def train_client(fl_iteration, client, x_target_test, y_target_test):
    
    if fl_iteration == 0:
        client.X_train,client.y_train = client.init_theta(client.x, client.y)
    
    client.train(client.X_train,client.y_train)
    client.train_acc = client.evaluate(client.x, client.y, acc=True)
    client.test_acc = client.evaluate(x_target_test, y_target_test, acc=True)

def output_DP_federated(model, X_train_size, epsilon_outDP, delta_outDP=1e-5):
    #gaussian mechanism 
    model.delta_outDP = delta_outDP
    sensitivity = 2/(X_train_size*model.lambda_) 
    sigma = np.sqrt(2 * np.log(1.25 / model.delta_outDP)) * (sensitivity / epsilon_outDP)
    model.theta_before_noise = model.theta
    model.theta = model.theta + np.random.normal(loc=0.0, scale=sigma, size=model.theta.shape)
