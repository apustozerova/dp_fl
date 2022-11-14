import torch
from torch import nn, optim


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

class Train_args():

    def __init__(self, learning_rate, weight_decay, epoch):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
         
def train_attack_model(model, train_data, train_target, train_args):

    optimizer = optim.Adam(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)
    
    model.train()
    optimizer.zero_grad()
    output = model(train_data)
    loss = nn.CrossEntropyLoss()(output, train_target.to(torch.long))
    loss.backward(retain_graph=True)
    optimizer.step()

    return model

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