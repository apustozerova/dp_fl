import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


class Train_args():

    def __init__(self, learning_rate, weight_decay, epoch):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
         
class Net_attack(nn.Module):

    def __init__(self, h_neurons, do, input_size):
        super(Net_attack, self).__init__()
        self.input_size = input_size
        self.h_neurons = h_neurons
        self.do = do
        self.fc1 = nn.Linear(input_size, h_neurons)
        self.fc2 = nn.Linear(h_neurons, 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(do)
        self.softmax = nn.Softmax(dim=1)   

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.softmax(x)
        return x

def train_attack_model(model, train_data, train_target, train_args):

    optimizer = optim.Adam(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)
    
    model.train()
    optimizer.zero_grad()
    output = model(train_data)
    loss = nn.CrossEntropyLoss()(output, train_target.to(torch.long))
    loss.backward(retain_graph=True)
    optimizer.step()

    return model

def train_shadow_models(number_of_sms,):

    for i in range(number_of_sms):
        batch_start = i*shadow_batch_size
        batch_end = (i+1)*shadow_batch_size
        
        shadow_model = algo.LogisticRegression_DPSGD()

        shadow_model.n_classes      = n_classes
        shadow_model.alpha          = 0.001
        shadow_model.max_iter       = 100*shadow_batch_size
        shadow_model.lambda_        = 10e-3
        shadow_model.tolerance      = 10e-5
        shadow_model.DP             = False

        X,y = shadow_model.init_theta(x_shadow_train[batch_start:batch_end], y_shadow_train[batch_start:batch_end] )
        shadow_model.SGD(X,y)
        print('Shadow model: ', i)
        shadow_model.evaluate(x_shadow_train[batch_start:batch_end], y_shadow_train[batch_start:batch_end])
        shadow_model.evaluate(x_shadow_test[batch_start:batch_end], y_shadow_test[batch_start:batch_end])
        s_ms[i] = shadow_model

def attack_evaluation(model, x, y, dev="cpu", extended=False):

    model.eval()

    with torch.no_grad():
        output =  model(x)
        out_target = output.argmax(1, keepdim=True)
        correct = out_target.to(dev).eq(y.to(dev).view_as(out_target.to(dev))).sum().item()
        acc = correct/y.shape[0]

        predicted_positive = output.argmax(1, keepdim=True) == 1
        labeled_positive = y == 1
        tp = predicted_positive.to(dev) * labeled_positive.to(dev).view_as(out_target)
        tp_count = tp.to(dev).sum().item()
        
        if predicted_positive.to(dev).sum().item() != 0:
            pre = tp_count / predicted_positive.to(dev).sum().item()
        else:
            pre = 0
        if labeled_positive.to(dev).sum().item() !=0:
            rec = tp_count / labeled_positive.to(dev).sum().item()
        else:
            rec = 0
    if extended:
        predicted_negative = output.argmax(1, keepdim=True) == 0
        labeled_negative = y == 0
        tn = predicted_negative.to(dev) * labeled_negative.to(dev).view_as(out_target)
        tn_count = tn.to(dev).sum().item()

        fp_count = predicted_positive.to(dev).sum().item() - tp.to(dev).sum().item()
        fn_count = labeled_positive.to(dev).sum().item() - tp.to(dev).sum().item()
        
        return acc, pre, rec, tp_count, tn_count, fp_count, fn_count
    else:
        return acc, pre, rec

def y_ohe(model, y):
    if len(np.unique(y)) > 2: #multi class classification
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(np.arange(model.n_classes).reshape(-1, 1))
        y = ohe.transform(y.reshape(-1,1)) #encoode the target values
    return y

def mi_attack_test(model, a_model, x_target_train, y_target_train, x_target_test, y_target_test):
    
    set_size = min(x_target_train.shape[0], x_target_test.shape[0])
    x_target_train, y_target_train =  x_target_train[:set_size], y_target_train[:set_size]
    x_target_test, y_target_test = x_target_test[:set_size], y_target_test[:set_size]

    train_pred = model.predict(x_target_train, y_target_train)
    test_pred = model.predict(x_target_test, y_target_test)

    y_train = y_ohe(model, y_target_train)
    y_test = y_ohe(model, y_target_test)

    # members
    labels = np.ones(x_target_train.shape[0])
    # non-members
    test_labels = np.zeros(x_target_test.shape[0])

    x_1 = np.concatenate((train_pred, test_pred))
    x_2 = np.concatenate((y_train, y_test))#.reshape((-1, 1))
    y_new = np.concatenate((labels, test_labels))

    attack_test_data = np.concatenate((x_1,x_2),axis=1)
    attack_test_target = y_new
    df = pd.DataFrame(attack_test_data)
    df['a_target'] = attack_test_target
    df = df.sample(frac = 1)

    attack_test_data = np.array(df.drop(['a_target'], axis=1))
    attack_test_target= np.array(df['a_target'])

    attack_test_data = torch.tensor(np.array(df.drop(['a_target'], axis=1)), dtype=torch.float, requires_grad=True)   
    attack_test_target = torch.tensor(np.array(df['a_target']), dtype=torch.float)
    
    test_acc, test_pre, test_rec = attack_evaluation(a_model, attack_test_data, attack_test_target)
    return test_acc, test_pre, test_rec

def data_shuffle(rand_seed, X_raw, y_raw):


    
    X_train, x_shadow, y_train, y_shadow = train_test_split(X_raw, y_raw, train_size=0.2, random_state=rand_seed)
#     print(X_train.shape, x_shadow.shape)

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
    return x_target_train, y_target_train, x_target_test, y_target_test

def test_mi_attack(attack_models, target_model, x_target_train, y_target_train, x_target_test, y_target_test):
    
    results = {}
    at_acc = []
    at_rec = []
    at_pre = []
    for am in attack_models:
        a_model = attack_models[am] 
        attack_acc, attack_pre, attack_rec = mi_attack_test(target_model, a_model, x_target_train, y_target_train, x_target_test, y_target_test)
        at_acc.append(attack_acc)
        at_pre.append(attack_pre)
        at_rec.append(attack_rec)
    results['attack_acc_mean'] = np.mean(at_acc)
    results['attack_acc_std'] = np.std(at_acc)
    results['attack_pre_mean'] = np.mean(at_pre)
    results['attack_pre_std'] = np.std(at_pre)
    results['attack_rec_mean'] = np.mean(at_rec)
    results['attack_rec_std'] = np.std(at_rec)
    
    return results    