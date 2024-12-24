import jax.numpy as jnp
from jax import random 
from jax import config
from scipy.interpolate import griddata
import os
import json

from models.DeepONet import *
from scripts.burgersUtils import *
from data.burgersDataGeneration import *

if __name__=="__main__":
    config_file = load_yaml_config('./config/burgersConfig.yaml')

    # Define hyperparameters and grid:
    loss_type = config_file["model"]["loss_type"]
    huber_delta = config_file["model"]["huber_delta"]
    kappa = config_file["global"]["kappa"] # corresponds to T
    period = int(config_file["global"]["period"])
    T_lim = int(config_file["global"]["T_lim"])

    # Size of rectangle
    x0 = int(config_file["global"]["x0"])
    y0 = int(config_file["global"]["y0"])

    # Intial condition
    sine_amplitude = float(config_file["train"]["sine_amplitude"])  # Amplitude of the sine terms

    # Training data
    m = config_file["model"]["branch_layers"][0]   # grid size in each dimention for discretizing the inhomogenuoes term, which mean that m is the branch net input dimention and it has to be a perfect square
    N_train_list = [int(eval(i)) for i in config_file["train"]["N_train"]]  # number of inhomogenuoes term candidates ( i.e f)
    P_train_list = [int(eval(i)) for i in  config_file["train"]["P_train"]]  # number of collocation points

    # Plotting data
    N_test = int(eval(config_file["plot"]["N_test"])) # number of test functions
    P_test = int(eval(config_file["plot"]["P_test"])) # number of test collocation points
    N_train_plot = int(eval(config_file["plot"]["N_train"])) # number of test functions
    P_train_plot = int(eval(config_file["plot"]["P_train"])) # number of test collocation points

    key = random.PRNGKey(1122) # a new unseen key
    keys = random.split(key, N_test)

    config.update("jax_enable_x64", True)
    f_test_vis, z_test_vis, u_test_vis = generate_test_data_visualization(key, P_test, x0, y0, T_lim, m, sine_amplitude)

    branch_layers = config_file["model"]["branch_layers"]
    trunk_layers =  config_file["model"]["trunk_layers"]
    model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=huber_delta)

    # Plot Rademacher
    file_name = f"gen_bound_{loss_type}_{huber_delta}" if loss_type=="huber" else f"gen_bound_{loss_type}"
    with open(f'./outputs/burgers/{file_name}.json', 'r') as json_file:
        save_dict = json.load(json_file)

    gen_error_list = np.array(save_dict['gen_error_list'])
    bound_list = np.array(save_dict['bound_list'])
    size_list = np.array(save_dict['size_list'])

    file_name = f"plot_burgers_{loss_type}_{huber_delta}_boundcorr" if loss_type=="huber" else f"plot_burgers_{loss_type}"
    plot_rademacher(gen_error_list, bound_list, size_list, file_name)