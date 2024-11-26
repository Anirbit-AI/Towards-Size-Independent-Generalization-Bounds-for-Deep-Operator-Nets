import jax.numpy as jnp
from jax import random, vmap
from jax import config
import json
import sys

sys.path.append("./")

from scripts.data.heatDataGeneration import *
from models.DeepONet import *
from scripts.heatUtils import *


if __name__=="__main__":
    config_file = load_yaml_config('./config/heatConfig.yaml')

    # Define hyperparameters and grid:
    loss_type = config_file["model"]["loss_type"]
    huber_delta = config_file["model"]["huber_delta"]
    T_lim = int(config_file["global"]["T_lim"]) # corresponds to T

    # Size of rectangle
    x0 = int(config_file["global"]["x0"])
    y0 = int(config_file["global"]["y0"])

    # Initial condition
    num_fourier_terms = int(config_file["train"]["num_fourier_terms"])  # Number of sine terms in the Fourier series
    sine_amplitude = float(config_file["train"]["sine_amplitude"])  # Amplitude of the sine terms

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
            key = random.PRNGKey(11)
            keys = random.split(key, N_train) # N keys to create N Functions

            config.update("jax_enable_x64", True)
            f_train, z_train, u_train = vmap(generate_one_training_data, (0, None, None, None, None, None, None))(keys, P_train, x0, y0, T_lim, m, sine_amplitude)

            # Reshape Data
            f_train = jnp.float32(f_train.reshape(N_train * P_train,-1))
            z_train = jnp.float32(z_train.reshape(N_train * P_train,-1))
            u_train = jnp.float32(u_train.reshape(N_train * P_train,-1))
            config.update("jax_enable_x64", False)

            # Test
            key = random.PRNGKey(4568) # different key than training data
            keys = random.split(key, N_test)

            config.update("jax_enable_x64", True)
            f_test1, z_test1, u_test1 = vmap(generate_one_test_data, (0, None, None, None, None, None, None))(keys, P_test, x0, y0, T_lim, m, sine_amplitude)

            # Reshape Data
            f_test = jnp.float32(f_test1.reshape(N_test * P_test,-1))
            z_test = jnp.float32(z_test1.reshape(N_test * P_test,-1))
            u_test = jnp.float32(u_test1.reshape(N_test * P_test,-1))

            config.update("jax_enable_x64", False)

            # Initialize model
            branch_layers = config_file["model"]["branch_layers"]
            trunk_layers =  config_file["model"]["trunk_layers"]
            model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=huber_delta)

            # Create dataset
            batch_size = 2**15 if 2**15 < P_train else P_train
            don_dataset = DataGenerator(f_train, z_train, u_train, batch_size)
            test_dataset = [f_test, z_test, u_test]

            # Train
            model.train(don_dataset, test_dataset, nIter=10000)

            # Save model params
            if(loss_type=="huber"):
                save_checkpoint(model.params, f'./outputs/heat/saved_models/model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}_{huber_delta}.npz')
            else:
                save_checkpoint(model.params, f'./outputs/heat/saved_models/model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}.npz')

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
    with open(f'./outputs/heat/{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file)