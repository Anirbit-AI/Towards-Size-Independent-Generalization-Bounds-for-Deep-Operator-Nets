import jax.numpy as jnp
from jax import random 
from jax import config
from scipy.interpolate import griddata
import os
import json

from models.DeepONet import *
from scripts.heatUtils import *
from data.heatDataGeneration import *

if __name__=="__main__":
    config_file = load_yaml_config('./config/heatConfig.yaml')

    # Define hyperparameters and grid:
    loss_type = config_file["model"]["loss_type"]
    huber_delta = config_file["model"]["huber_delta"]
    T_lim = int(config_file["global"]["T_lim"]) # corresponds to T

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

    for P_train in P_train_list:
        for N_train in N_train_list:
            # Predict
            model_name = f"model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}_{huber_delta}" if loss_type=="huber" else f"model_N_train_{N_train}_P_train_{P_train}_checkpoint_{loss_type}"
            params = load_checkpoint(f"./outputs/heat/saved_models/{model_name}.npz")
            u_pred = jnp.zeros((11,P_test))

            # Predict
            for i in range(11):
                u_pred = u_pred.at[i,:].set(model.predict_u(params, f_test_vis, z_test_vis[i,:,:])[:,0])

            # Generate an uniform mesh
            x = jnp.linspace(0, x0, 121)
            y = jnp.linspace(0, y0, 121)
            XX, YY = jnp.meshgrid(x, y)

            time_steps = jnp.linspace(0, T_lim, num=(int(P_test**0.5)))
            # Grid data
            U_pred = jnp.zeros((11,121,121))
            U_test = jnp.zeros((11,121,121))
            for i in range(11):
                U_pred = U_pred.at[i].set(griddata(z_test_vis[i,:,:2], u_pred[i,:].flatten(), (XX,YY), method='cubic'))
                U_test = U_test.at[i].set(griddata(z_test_vis[i,:,:2], u_test_vis[i,:].flatten(), (XX,YY), method='cubic'))

            for ts in range(11):
                plot_actual_pred(XX, YY, U_test, U_pred, time_steps,ts, loss_type)

    file_name = f"gen_bound_{loss_type}_{huber_delta}" if loss_type=="huber" else f"gen_bound_{loss_type}"
    with open(f'./outputs/heat/{file_name}.json', 'r') as json_file:
        save_dict = json.load(json_file)

    gen_error_list = np.array(save_dict['gen_error_list'])
    bound_list = np.array(save_dict['bound_list'])
    size_list = np.array(save_dict['size_list'])

    file_name = f"plot_heat_{loss_type}_{huber_delta}_boundcorr" if loss_type=="huber" else f"plot_heat_{loss_type}"
    plot_rademacher(gen_error_list, bound_list, size_list, file_name)

    if(f"model_N_train_{N_train_plot}_P_train_{P_train_plot}_checkpoint_huber_{huber_delta}.npz" in os.listdir("./outputs/heat/saved_models") or f"model_N_train_{N_train_plot}_P_train_{P_train_plot}_checkpoint_l2.npz" in os.listdir("./outputs/heat/saved_models")):
        try:
            U_pred_huber = plot_predict(model, P_test, f_test_vis, z_test_vis, x0, y0, N_train, P_train, "huber", huber_delta)
        except ValueError:
            print("Checkpoints for huber loss does not exist")
        
        try:
            U_pred_l2 = plot_predict(model, P_test, f_test_vis, z_test_vis, x0, y0, N_train, P_train, "l2")
        except ValueError:
            print("Checkpoints for l2 loss does not exist")

        for ts in range(5):
            plot_both_losses(XX, YY, U_test, U_pred_huber, U_pred_l2, time_steps, ts)