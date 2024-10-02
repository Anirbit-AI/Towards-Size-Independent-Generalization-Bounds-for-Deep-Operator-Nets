import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import configparser

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def save_checkpoint(params, filename):
    # Flatten parameters
    flat_params, unravel_fn = ravel_pytree(params)

    # Save the flattened parameters and the metadata
    np.savez(filename, params=flat_params, unravel_fn=unravel_fn)

def load_checkpoint(filepath):
    # Load the checkpoint 
    data = np.load(filepath, allow_pickle=True)

    # Unravel function 
    unravel_fn = data['unravel_fn'].item()  # Convert back to function
    flat_params = data['params']

    # Reconstruct the parameters
    params = unravel_fn(flat_params)
    return params

def calculate_radbound(model, N_train, P_train):
    # C value computation
    branch_params, trunk_params = model.params
    C_vals = []
    n = len(branch_params)

    # Weight matrices are transposed for this specific implementation (y = XW)
    b_n1 = branch_params[n-2].T.shape[0]
    t_n1 = trunk_params[n-2].T.shape[0]
    p = branch_params[n-1].T.shape[0]

    # Spectral Norm computation
    branch_spec_norms = []
    trunk_spec_norms = []

    for i in range(n):
        branch_spec_norms.append(np.linalg.norm(branch_params[i].T, ord=2).item())
        trunk_spec_norms.append(np.linalg.norm(trunk_params[i].T, ord=2).item())


    norm_matrix_n = np.abs(np.sum(np.array([np.outer((branch_params[n-1].T)[j], (trunk_params[n-1].T)[j]) for j in range(p)]), axis=0))
    norm_vec_B = np.array([np.linalg.norm(np.array([(branch_params[n-2].T)[k1]])) for k1 in range(b_n1)])
    norm_vec_T = np.array([np.linalg.norm(np.array([(trunk_params[n-2].T)[k2]])) for k2 in range(t_n1)])

    C_1 = (norm_vec_B.T @ norm_matrix_n @ norm_vec_T).item()
    C_vals.append(C_1)

    for k in range(2, n):
        b_k = branch_params[n-k-1].T.shape[0]
        t_k = trunk_params[n-k-1].T.shape[0]
        norm_matrix_k = np.array([[np.linalg.norm((branch_params[n-k-1].T)[j1])*np.linalg.norm((trunk_params[n-k-1].T)[j2]) for j2 in range(t_k)] for j1 in range(b_k)])
        C_k = np.linalg.norm(norm_matrix_k, ord=2).item()
        C_vals.append(C_k)

    bound = (C_vals[0]*C_vals[1])/np.sqrt(N_train*P_train)

    return bound

def plot_predict(model, P_test, f_test_vis, z_test_vis, x0, y0, loss_type):
    # Predict - both huber and l2
    params = load_checkpoint(f"./outputs/saved_models/model_checkpoint_{loss_type}.npz")
    u_pred = jnp.zeros((11,P_test))

    # Predict
    for i in range(11):
        u_pred = u_pred.at[i,:].set(model.predict_u(params, f_test_vis, z_test_vis[i,:,:])[:,0])

    # Generate an uniform mesh
    x = jnp.linspace(0, x0, 121)
    y = jnp.linspace(0, y0, 121)
    XX, YY = jnp.meshgrid(x, y)

    # Grid data
    U_pred = jnp.zeros((11,121,121))
    for i in range(11):
        U_pred = U_pred.at[i].set(griddata(z_test_vis[i,:,:2], u_pred[i,:].flatten(), (XX,YY), method='cubic'))

    return U_pred

# 3D plot function
def plot_3d(ax, X, Y, u):
    ax.plot_surface(X, Y, u, cmap='plasma')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y,t)$')

# Color plot function
def plot(ax, X, Y, u):
    pcm = ax.pcolor(X, Y, u, cmap='plasma')
    plt.colorbar(pcm, ax=ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

# Error plot function
def plot_us(x,u,y,s):
  fig, ax1 = plt.subplots(figsize=(8, 6))
  plt.rcParams['font.size'] = '18'
  color='#440154'
  wdt=1.5
  ax1.plot(x,u,'k--',label='$u(x)=ds/dx$',linewidth=wdt)
  ax1.plot(y,s,'-',label='$s(x)=s(0)+\int u(t)dt|_{t=y}$',linewidth=wdt)
  ax1.set_xlabel('x')
  ax1.set_ylabel('u')
  ax1.tick_params(axis='y', color=color)
  ax1.legend(loc = 'lower right', ncol=1)

def plot_train_test_error(model, filename):
    # Visualizations
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Colors from the 'viridis' colormap
    medium_purple = '#5B278F'
    light_purple = '#3b528b'
    viri_green = '#00A896'

    # Total loss per 100 iteration
    total_loss_eval_numbers = range(1, len(model.loss_don_log) + 1)
    axs[0].plot(total_loss_eval_numbers, model.loss_don_log, '--', color=medium_purple, label='Training loss')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[0].set_ylabel('Training Loss', fontsize='large')
    axs[0].set_title('Evolution of Training Loss Over Iterations', fontsize='large')


    # Test loss
    test_loss_eval_numbers = range(1, len(model.loss_test_log) + 1)
    axs[1].plot(test_loss_eval_numbers, model.loss_test_log, '--', color=medium_purple, label='Test loss')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[1].set_ylabel('Test Loss', fontsize='large')
    axs[1].set_title('Evolution of Test Loss Over Iterations', fontsize='large')

    # Average fractional test loss
    AFTL_eval_numbers = range(1, len(model.loss_AF_test_log) + 1)
    axs[2].plot(AFTL_eval_numbers, model.loss_AF_test_log, '--', color=medium_purple, label='Average Fractional Test loss')
    axs[2].set_yscale('log')
    axs[2].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[2].set_ylabel('Average Fractional Test Loss', fontsize='large')
    axs[2].set_title('Evolution of Average Fractional Test Loss over Iterations', fontsize='large')

    plt.savefig(f"./outputs/train_test/{filename}.png", bbox_inches ="tight")

    return

def plot_actual_pred(XX, YY, U_test, U_pred, time_steps, ts, loss_type):
   # Create a new figure with two rows of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: 3D plots
    # Analytical Solution of the Heat Equation
    axs[0, 0] = fig.add_subplot(221, projection='3d')
    plot_3d(axs[0, 0], XX, YY, U_test[ts,:,:])
    axs[0, 0].set_title(f"Analytical Solution - Heat Equation at t = {time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet
    axs[0, 1] = fig.add_subplot(222, projection='3d')
    plot_3d(axs[0, 1], XX, YY, U_pred[ts,:,:])
    axs[0, 1].set_title(f"Predicted Solution - DeepONet at t ={time_steps[ts]:.1f}", fontsize='large')

    # Bottom row: 2D color plots
    # Analytical Solution of the Heat Equation
    plot(axs[1, 0], XX, YY, U_test[ts,:,:])
    axs[1, 0].set_title(f"Analytical Solution - Heat Equation at t = {time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet
    plot(axs[1, 1], XX, YY, U_pred[ts,:,:])
    axs[1, 1].set_title(f"Predicted Solution - DeepONet at t = {time_steps[ts]:.1f}", fontsize='large')

    plt.savefig(f"./outputs/actual_predict/actual_predicted_plots_timestep_{ts}_loss_type_{loss_type}.png", bbox_inches ="tight")

    return

def plot_both_losses(XX, YY, U_test, U_pred_huber, U_pred_l2, time_steps, ts):
   # Create a new figure with two rows of subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Top row: 3D plots
    # Analytical Solution of the Heat Equation
    axs[0, 0] = fig.add_subplot(231, projection='3d')
    plot_3d(axs[0, 0], XX, YY, U_test[ts,:,:])
    axs[0, 0].set_title(f"Analytical Solution - Heat Equation at t = {time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet l2
    axs[0, 1] = fig.add_subplot(232, projection='3d')
    plot_3d(axs[0, 1], XX, YY, U_pred_l2[ts,:,:])
    axs[0, 1].set_title(f"Predicted Solution - DeepONet and l2 at t ={time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet huber
    axs[0, 2] = fig.add_subplot(233, projection='3d')
    plot_3d(axs[0, 2], XX, YY, U_pred_huber[ts,:,:])
    axs[0, 2].set_title(f"Predicted Solution - DeepONet and Huber at t ={time_steps[ts]:.1f}", fontsize='large')

    # Bottom row: 2D color plotss
    # Analytical Solution of the Heat Equation
    plot(axs[1, 0], XX, YY, U_test[ts,:,:])
    axs[1, 0].set_title(f"Analytical Solution - Heat Equation at t = {time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet l2
    plot(axs[1, 1], XX, YY, U_pred_l2[ts,:,:])
    axs[1, 1].set_title(f"Predicted Solution - DeepONet and l2 at t = {time_steps[ts]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet huber
    plot(axs[1, 2], XX, YY, U_pred_huber[ts,:,:])
    axs[1, 2].set_title(f"Predicted Solution - DeepONet and Huber at t = {time_steps[ts]:.1f}", fontsize='large')

    plt.savefig(f"./outputs/actual_predict/actual_predicted_plots_timestep_{ts}_loss_type_both.png", bbox_inches ="tight")

    return