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

