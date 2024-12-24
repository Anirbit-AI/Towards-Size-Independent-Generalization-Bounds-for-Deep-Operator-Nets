import jax.numpy as jnp
from jax import random, vmap
from jax import config
import json

from data.burgersDataGeneration import *
from models.DeepONet import *
from scripts.burgersUtils import *


if __name__=="__main__":
    config_file = load_yaml_config('./config/heatConfig.yaml')

    # Define hyperparameters and grid:
    loss_type = config_file["model"]["loss_type"]
    huber_delta = config_file["model"]["huber_delta"]
    kappa = config_file["global"]["kappa"] # corresponds to T
    period = int(config_file["global"]["period"])
    T_lim = int(config_file["global"]["T_lim"])

    # Initial condition
    num_sine_terms = int(config_file["train"]["num_sine_terms"])  # Number of sine terms in the Fourier series
    sine_amplitude = float(config_file["train"]["sine_amplitude"])  # Amplitude of the sine terms

    # Resolution of the solution (Grid of 100x100)
    Nx = config_file["global"]["Nx"]
    Nt = config_file["global"]["Nt"]
    
    # Training data
    m = config_file["model"]["branch_layers"][0]   # grid size in each dimention for discretizing the inhomogenuoes term, which mean that m is the branch net input dimention and it has to be a perfect square
    N_train_list = [int(eval(i)) for i in config_file["train"]["N_train"]]  # number of inhomogenuoes term candidates ( i.e f)
    P_train_list = [int(eval(i)) for i in  config_file["train"]["P_train"]]  # number of collocation points

    # Test data
    N_test = int(eval(config_file["test"]["N_test"])) # number of test functions
    P_test = int(eval(config_file["test"]["P_test"])) # number of test collocation points

    bound_list = []
    gen_error_list = []

    for P_train in P_train_list:
        for N_train in N_train_list:
            'Data Generation'
            key = random.PRNGKey(0)
            keys = random.split(key, N_train)

            config.update("jax_enable_x64", True)
            u_train, y_train_don, s_train_don = vmap(generate_one_training_data, (0, None, None, None, None, None, None, None, None, None))(keys, P_train, num_sine_terms, sine_amplitude, Nx , Nt, T_lim, period, kappa, m)

            # Reshape Data
            u_don_train = np.float32(u_train.reshape(N_train * P_train,-1))
            y_don_train = np.float32(y_train_don.reshape(N_train * P_train,-1))
            s_don_train = np.float32(s_train_don.reshape(N_train * P_train,-1))
            config.update("jax_enable_x64", False)

            # Test
            key = random.PRNGKey(12345) # different key than training data
            keys = random.split(key, N_test)
            config.update("jax_enable_x64", False)

            u_test, y_test, s_test = vmap(generate_one_test_data, (0, None, None, None, None, None, None, None))(keys, P_test, num_sine_terms, sine_amplitude, T_lim, period, kappa, m)

            #Reshape Data
            u_test = np.float32(u_test.reshape(N_test * P_test**2,-1))
            y_test = np.float32(y_test.reshape(N_test * P_test**2,-1))
            s_test = np.float32(s_test.reshape(N_test * P_test**2,-1))

            config.update("jax_enable_x64", False)

            # Initialize model
            branch_layers = config_file["model"]["branch_layers"]
            trunk_layers =  config_file["model"]["trunk_layers"]
            model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=huber_delta, activation=jnp.abs)

            # Create dataset
            batch_size = 2**15 if 2**15 < P_train else P_train
            don_dataset = DataGenerator(u_don_train, y_don_train, s_don_train, batch_size)
            test_dataset = [u_test, y_test, s_test]

            # Train
            model.train(don_dataset, test_dataset, nIter=10000)

            # Save model params
            if(loss_type=="huber"):
                save_checkpoint(model.params, f'./outputs/burgers/saved_models/model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}_{huber_delta}.npz')
            else:
                save_checkpoint(model.params, f'./outputs/burgers/saved_models/model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}.npz')

            # Plot train and test errors
            if(loss_type=="huber"):
                plot_train_test_error(model, f"train_test_error_plots_N_train_{N_train}_P_train_{P_train}_{loss_type}_{huber_delta}")
            else:
                plot_train_test_error(model, f"train_test_error_plots_N_train_{N_train}_P_train_{P_train}_{loss_type}")

            # Calculate the Rademacher bound
            bound_list.append(calculate_radbound(model, N_train, P_train))

            # Generalization Error
            gen_error = np.abs(model.loss_don_log[-1] - model.loss_test_log[-1], dtype=np.float64)
            gen_error_list.append(gen_error)
        
    size_list = np.array(N_train_list)*np.array(P_train_list)

    save_dict = {"bound_list" : bound_list, "gen_error_list" : gen_error_list, "size_list" : size_list.tolist(), "N_train" : N_train_list, "P_train" : P_train_list}
    file_name = f"gen_bound_{loss_type}_{huber_delta}" if loss_type=="huber" else f"gen_bound_{loss_type}"
    with open(f'./outputs/burgers/{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file)