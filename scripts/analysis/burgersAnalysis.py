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
    Nx = int(config_file["global"]["Nx"])
    Nt = int(config_file["global"]["Nt"])

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
    u_test_vis, y_test_vis, s_test_vis = generate_test_data_visualization(keys, P_test, x0, y0, T_lim, m, sine_amplitude)

    branch_layers = config_file["model"]["branch_layers"]
    trunk_layers =  config_file["model"]["trunk_layers"]
    model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=huber_delta)

    for P_train in P_train_list:
        for N_train in N_train_list:
            # Predict
            model_name = f"model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}_{huber_delta}" if loss_type=="huber" else f"model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}"
            params = load_checkpoint(f"./outputs/burgers/saved_models/{model_name}.npz")
            s_pred = model.predict_u(params, u_test_vis, y_test_vis)

            # Generate an uniform mesh
            x = jnp.linspace(-period*jnp.pi, period*jnp.pi, Nx)
            t = jnp.linspace(0, T_lim, Nt)
            XX, TT = jnp.meshgrid(x, t)

            # Grid data
            S_pred = griddata(y_test_vis, s_pred.flatten(), (XX,TT), method='cubic')
            S_test = griddata(y_test_vis, s_test_vis.flatten(), (XX,TT), method='cubic')

            plot_actual_pred(XX, TT, S_test, S_pred, loss_type)

    # Plot Rademacher
    file_name = f"gen_bound_{loss_type}_{huber_delta}" if loss_type=="huber" else f"gen_bound_{loss_type}"
    with open(f'./outputs/burgers/{file_name}.json', 'r') as json_file:
        save_dict = json.load(json_file)

    gen_error_list = np.array(save_dict['gen_error_list'])
    bound_list = np.array(save_dict['bound_list'])
    size_list = np.array(save_dict['size_list'])

    file_name = f"plot_burgers_{loss_type}_{huber_delta}_boundcorr" if loss_type=="huber" else f"plot_burgers_{loss_type}"
    plot_rademacher(gen_error_list, bound_list, size_list, file_name)