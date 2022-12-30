
def aggregate(clients):
    
    global_model = clients[0].theta 
    for i in range(1, len(clients.keys())):
        global_model += clients[i].theta 

    return global_model/len(clients.keys())

def update_clients(clients, global_model):
    
    for i in clients:
        clients[i].theta = global_model


def train_client(client, x_target_test, y_target_test):
    
    X,y = client.init_theta(client.x, client.y)
    client.train(X,y)
    client.train_acc = client.evaluate(client.x, client.y, acc=True)
    client.test_acc = client.evaluate(x_target_test, y_target_test, acc=True)
