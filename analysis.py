import jax.numpy as jnp
from jax import random 
from jax import config
from scipy.interpolate import griddata

from scripts.DeepONet import *
from scripts.utils import *
from scripts.data_generation import *

if __name__=="__main__":
    config_file = load_config('config.ini')

    # Define hyperparameters and grid:
    loss_type = config_file["MODEL"]["loss_type"]
    T_lim = int(config_file["GLOBAL"]["T_lim"]) # corresponds to T

    # Size of rectangle
    x0 = int(config_file["GLOBAL"]["x0"])
    y0 = int(config_file["GLOBAL"]["y0"])

    # Intial condition
    sine_amplitude = float(config_file["TRAIN"]["sine_amplitude"])  # Amplitude of the sine terms

    # Training data
    m = int(config_file["MODEL"]["branch_layers"].split(",")[0])   # grid size in each dimention for discretizing the inhomogenuoes term, which mean that m is the branch net input dimention and it has to be a perfect square

    # Plotting data
    N_test = int(config_file["PLOT"]["N_test"]) # number of test functions
    P_test = int(config_file["PLOT"]["P_test"]) # number of test collocation points

    key = random.PRNGKey(1122) # a new unseen key
    keys = random.split(key, N_test)

    config.update("jax_enable_x64", True)
    f_test_vis, z_test_vis, u_test_vis = generate_test_data_visualization(key, P_test, x0, y0, T_lim, m, sine_amplitude)

    branch_layers = [ int(i.strip()) for i in config_file["MODEL"]["branch_layers"].split(",")]
    trunk_layers =  [ int(i.strip()) for i in config_file["MODEL"]["trunk_layers"].split(",")]
    model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=0.5**9)

    # Predict
    params = load_checkpoint(f"./outputs/saved_models/model_checkpoint_{loss_type}.npz")
    # params = model.get_params(model.opt_state)
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