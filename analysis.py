import jax.numpy as jnp
from jax import random 
from jax import config
from scipy.interpolate import griddata

from scripts.DeepONet import *
from scripts.utils import *
from scripts.data_generation import *

if __name__=="__main__":
    # Define hyperparameters and grid:
    kappa = 1
    T_lim = 1 # corresponds to T

    #Size of rectangle
    x0 = 1
    y0 = 1

    # Initial condition
    num_fourier_terms = 2  # Number of sine terms in the Fourier series
    sine_amplitude = 0.2  # Amplitude of the sine terms

    # Training data
    m = 100   # grid size in each dimention for discretizing the inhomogenuoes term, which mean that m is the branch net input dimention and it has to be a perfect square
    N_train = 2**9  # number of inhomogenuoes term candidates ( i.e f)
    P_train = 4**6 # number of evaluation points for each training loss


    P_test = 121
    N_test = 1
    key = random.PRNGKey(1122) # a new unseen key
    keys = random.split(key, N_test)

    config.update("jax_enable_x64", True)
    f_test_vis, z_test_vis, u_test_vis = generate_test_data_visualization(key, P_test, x0, y0, T_lim, m)

    branch_layers = [m, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    trunk_layers =  [3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    model = DeepONet(branch_layers, trunk_layers, loss_type= "l2", huber_delta=0.5**9)

    # Predict
    params = load_checkpoint("./outputs/saved_models/model_checkpoint.npz")
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

    plot_actual_pred(XX, YY, U_test, U_pred, time_steps)