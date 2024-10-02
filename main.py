import jax.numpy as jnp
from jax import random, vmap
from jax import config

from scripts.data_generation import *
from scripts.DeepONet import *
from scripts.utils import *


if __name__=="__main__":
    config_file = load_config('config.ini')

    # Define hyperparameters and grid:
    loss_type = config_file["MODEL"]["loss_type"]
    huber_delta = eval(config_file["MODEL"]["huber_delta"])
    T_lim = int(config_file["GLOBAL"]["T_lim"]) # corresponds to T

    #Size of rectangle
    x0 = int(config_file["GLOBAL"]["x0"])
    y0 = int(config_file["GLOBAL"]["y0"])

    # Initial condition
    num_fourier_terms = int(config_file["TRAIN"]["num_fourier_terms"])  # Number of sine terms in the Fourier series
    sine_amplitude = float(config_file["TRAIN"]["sine_amplitude"])  # Amplitude of the sine terms

    # Training data
    m = int(config_file["MODEL"]["branch_layers"].split(",")[0])   # grid size in each dimention for discretizing the inhomogenuoes term, which mean that m is the branch net input dimention and it has to be a perfect square
    N_train = eval(config_file["TRAIN"]["N_train"])  # number of inhomogenuoes term candidates ( i.e f)
    P_train = eval(config_file["TRAIN"]["P_train"]) # number of evaluation points for each training loss

    # Test data
    N_test = eval(config_file["TEST"]["N_test"]) # number of test functions
    P_test = eval(config_file["TEST"]["P_test"]) # number of test collocation points

    'Data Generation'
    key = random.PRNGKey(11)
    keys = random.split(key, N_train) # N keys to create N Functions

    config.update("jax_enable_x64", True)
    f_train, z_train, u_train = vmap(generate_one_training_data, (0, None, None, None, None, None, None))(keys, P_train, x0, y0, T_lim, m, sine_amplitude)

    #Reshape Data
    f_train = jnp.float32(f_train.reshape(N_train * P_train,-1))
    z_train = jnp.float32(z_train.reshape(N_train * P_train,-1))
    u_train = jnp.float32(u_train.reshape(N_train * P_train,-1))
    config.update("jax_enable_x64", False)

    # Test
    key = random.PRNGKey(4568) # different key than training data
    keys = random.split(key, N_test)

    config.update("jax_enable_x64", True)
    f_test1, z_test1, u_test1 = vmap(generate_one_test_data, (0, None, None, None, None, None, None))(keys, P_test, x0, y0, T_lim, m, sine_amplitude)

    #Reshape Data
    f_test = jnp.float32(f_test1.reshape(N_test * P_test,-1))
    z_test = jnp.float32(z_test1.reshape(N_test * P_test,-1))
    u_test = jnp.float32(u_test1.reshape(N_test * P_test,-1))

    config.update("jax_enable_x64", False)

    # Initialize model
    branch_layers = [ int(i.strip()) for i in config_file["MODEL"]["branch_layers"].split(",")]
    trunk_layers =  [ int(i.strip()) for i in config_file["MODEL"]["trunk_layers"].split(",")]
    model = DeepONet(branch_layers, trunk_layers, loss_type=loss_type, huber_delta=huber_delta, activation=jnp.abs)

    # Create dataset
    batch_size = 2**15
    don_dataset = DataGenerator(f_train, z_train, u_train, batch_size)
    test_dataset = [f_test, z_test, u_test]

    # Train
    model.train(don_dataset, test_dataset, nIter=1000)

    # Plot train and test errors
    plot_train_test_error(model, f"train_test_error_plots_{loss_type}")

    # Calculate the Rademacher bound
    bound_list = []
    bound_list.append(calculate_radbound(model, N_train, P_train))
    np.save('./outputs/gen_bound_{loss_type}.npy', bound_list)

    # Save model params
    save_checkpoint(model.params, f'./outputs/saved_models/model_checkpoint_{loss_type}.npz')