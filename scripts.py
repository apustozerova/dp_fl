import numpy as np
import algo

def set_model_params(model, params):

    model.n_classes      = params['n_classes']
    model.alpha          = params['alpha']
    model.max_iter       = params['max_iter']
    model.lambda_        = params['lambda_']
    model.tolerance      = params['tolerance']
    model.sgdDP          = params['sgdDP']
    model.L              = params['L']
    model.C              = params['C']
    model.epsilon        = params['epsilon']
    model.delta          = params['delta']

def output_DP(model,  X_train_size, epsilon_out_DP, delta_out_DP=1e-5):
    #gaussian mechanism 
    noise_model = algo.LogisticRegression_DPSGD()
    set_model_params(noise_model, model.__dict__)
    
    noise_model.DP_out = True
    noise_model.epsilon_out = epsilon_out_DP
    noise_model.delta_out = delta_out_DP
    noise_model.sensitivity = 2/(X_train_size*model.lambda_) 

    sigma = np.sqrt(2 * np.log(1.25 / noise_model.delta_out)) * (noise_model.sensitivity / noise_model.epsilon_out)

    ns=np.random.normal(loc=0.0, scale=sigma, size=model.theta.shape)

    noise_model.theta = model.theta + ns
    
    return noise_model

def load_purchase(rand_seed):
    x_target_train = np.load('data/rs'+str(rand_seed)+'_x_target_train.npy')
    y_target_train = np.load('data/rs'+str(rand_seed)+'_y_target_train.npy')
    x_target_test = np.load('data/rs'+str(rand_seed)+'_x_target_test.npy')
    y_target_test = np.load('data/rs'+str(rand_seed)+'_y_target_test.npy')

    return x_target_train, y_target_train, x_target_test, y_target_test

def load_loan(rand_seed, tr_size):

    x_target_train = np.load(f'data/loan_rs{rand_seed}_size{tr_size}_xtrain.npy')
    y_target_train = np.load(f'data/loan_rs{rand_seed}_size{tr_size}_ytrain.npy')
    x_target_test = np.load(f'data/loan_rs{rand_seed}_size{tr_size}_xtest.npy')
    y_target_test = np.load(f'data/loan_rs{rand_seed}_size{tr_size}_ytest.npy')

    return x_target_train, y_target_train, x_target_test, y_target_test

def load_texas():

    x_target_train = np.load(f'data/texas_x_target_train.npy')
    y_target_train = np.load(f'data/texas_y_target_train.npy')
    x_target_test = np.load(f'data/texas_x_target_test.npy')
    y_target_test = np.load(f'data/texas_y_target_test.npy')

    return x_target_train, y_target_train, x_target_test, y_target_test
