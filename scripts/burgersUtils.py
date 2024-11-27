import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
import configparser
import yaml


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # Use safe_load for security
    return config

# def load_config(file_path):
#     config = configparser.ConfigParser()
#     config.read(file_path)
#     return config

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

# 3D plot function
def plot_3d(ax, X, T, f):
    surf = ax.plot_surface(X, T, f, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$s(x,t)$')

# Color plot function
def plot(ax, X, T, f):
    pcm = ax.pcolor(X, T, f, cmap='viridis')
    plt.colorbar(pcm, ax=ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')

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

# Error plot function
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

    plt.savefig(f"./outputs/heat/train_test/{filename}.png", bbox_inches ="tight")

    return

def plot_rademacher(gen_error_list, bound_list, size_list, file_name):
    sb.set_theme(style='whitegrid', palette="deep", font_scale=1.1, rc={"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})

    with sb.color_palette('viridis_r', 7):
        fig = plt.figure(figsize=(9, 6))

    plt.xlabel('Generalization Error Bound')
    plt.ylabel('|Train-Error - Test-Error|')

    corr = np.corrcoef(np.array(bound_list), np.array(gen_error_list))[0][1]
    g = sb.scatterplot(x=bound_list, y=gen_error_list, edgecolor='black', linewidth=1.2, s=35)#, label='$L_2$ loss (Correlation = {:.3f})'.format(corr))
    for i, x in enumerate(size_list):
        plt.annotate(str("%.1e" % x), (bound_list[i], gen_error_list[i]), xytext=(2, 2), textcoords='offset points')

    plt.annotate('Correlation Coefficient = {:.3f}'.format(corr), xy=(0.03, 0.92), xycoords='axes fraction', color='black',bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5', alpha=0.6))

    # Add a black boundary around the graph
    plt.gca().spines['top'].set_edgecolor('black')
    plt.gca().spines['top'].set_linewidth(1.2)
    plt.gca().spines['right'].set_edgecolor('black')
    plt.gca().spines['right'].set_linewidth(1.2)
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['bottom'].set_linewidth(1.2)
    plt.gca().spines['left'].set_edgecolor('black')
    plt.gca().spines['left'].set_linewidth(1.2)

    plt.tick_params(axis='x', which='both', bottom=True, top=False, length=6, width=1.5)
    plt.tick_params(axis='y', which='both', left=True, right=False, length=6, width=1.5)

    # Format the ticks to scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    # Enable scientific notation for the x and y axes
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.savefig(f"./outputs/heat/{file_name}.svg")
    plt.savefig(f"./outputs/heat/{file_name}.png", bbox_inches ="tight")

def plot_actual_pred(XX, TT, S_test, S_pred):
    # Create a new figure with two rows of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    # Top row: 3D plots
    # Numerical Solution of Burgers' Equation
    # axs[0, 0].remove()  # Remove the existing 2D subplot
    axs[0, 0].tick_params(labelbottom=False, labelleft=False)
    axs[0, 0] = fig.add_subplot(221, projection='3d')
    plot_3d(axs[0, 0], XX, TT, S_test)
    axs[0, 0].set_title("Numerical Solution of Burgers' Equation", fontsize=16)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

    # Predicted Solution using DeepONet
    # axs[0, 1].remove()  # Remove the existing 2D subplot
    axs[0, 1].tick_params(labelbottom=False, labelleft=False)
    axs[0, 1] = fig.add_subplot(222, projection='3d')
    plot_3d(axs[0, 1], XX, TT, S_pred)
    axs[0, 1].set_title("Predicted Solution using DeepONet", fontsize=16)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

    # Bottom row: 2D color plots
    # Numerical Solution of Burgers' Equation
    plot(axs[1, 0], XX, TT, S_test)
    axs[1, 0].set_title("Numerical Solution of Burgers' Equation", fontsize=16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

    # Predicted Solution using DeepONet
    plot(axs[1, 1], XX, TT, S_pred)
    axs[1, 1].set_title("Predicted Solution using DeepONet", fontsize=16)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

    plt.savefig(f"./outputs/heat/plot_actual_pred.png", bbox_inches ="tight")