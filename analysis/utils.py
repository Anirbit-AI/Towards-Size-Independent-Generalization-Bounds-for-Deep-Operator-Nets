import jax.numpy as jnp


def generate_test_data_visualization(key, P_test):

    # Sample collocation points
    x_test = jnp.linspace(0, x0, num=(int(P_test**0.5)))
    y_test = jnp.linspace(0, y0, num=(int(P_test**0.5)))
    t_test = jnp.linspace(0, T_lim, num=(int(P_test**0.5)))
    x_test_mesh, y_test_mesh = jnp.meshgrid(x_test, y_test)
    x_test = x_test_mesh.flatten().reshape(-1,1)
    y_test = y_test_mesh.flatten().reshape(-1,1)

    # Pre-allocate the array for efficiency
    z_test = jnp.zeros((len(t_test),P_test, 3))  # Shape: (number of time steps, 3 columns)

    # Testing collocation points
    for i, t in enumerate(t_test):
      # Create a single coordinate pair for each time step
      coordinates = jnp.hstack([x_test, y_test, t * jnp.ones_like(x_test)])
      z_test = z_test.at[i, :, :].set(coordinates)


    # Input sensor locations and measurements
    x_sensor = jnp.linspace(0, x0, int(m**0.5), endpoint = True)
    y_sensor = jnp.linspace(0, y0, int(m**0.5), endpoint = True)

    # Generate initial condition
    f, c_m, d_n = f_testing(2, key)
    f_test = jnp.tile(f(x_sensor,y_sensor).T, (P_test,1))

    u_test = jnp.zeros((len(t_test),P_test))  # Shape: (number of time steps, 3 columns)
    for i in range(len(t_test)):
      u_test = u_test.at[i,:].set(u(A_mn(c_m, d_n), z_test[i,:,0],z_test[i,:,1],z_test[i,:,2]))

    return f_test, z_test, u_test

def plot_analytic_pred():
   # Create a new figure with two rows of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: 3D plots
    # Analytical Solution of the Heat Equation
    axs[0, 0] = fig.add_subplot(221, projection='3d')
    plot_3d(axs[0, 0], XX, YY, U_test[0,:,:])
    axs[0, 0].set_title(f"Analytical Solution of the Heat Equation at t = {time_steps[0]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet
    axs[0, 1] = fig.add_subplot(222, projection='3d')
    plot_3d(axs[0, 1], XX, YY, U_pred[0,:,:])
    axs[0, 1].set_title(f"Predicted Solution using DeepONet at t ={time_steps[0]:.1f}", fontsize='large')

    # Bottom row: 2D color plots
    # Analytical Solution of the Heat Equation
    plot(axs[1, 0], XX, YY, U_test[0,:,:])
    axs[1, 0].set_title(f"Analytical Solution of the Heat Equation at t = {time_steps[0]:.1f}", fontsize='large')

    # Predicted Solution using DeepONet
    plot(axs[1, 1], XX, YY, U_pred[0,:,:])
    axs[1, 1].set_title(f"Predicted Solution using DeepONet at t = {time_steps[0]:.1f}", fontsize='large')

    plt.tight_layout()
    plt.show()
