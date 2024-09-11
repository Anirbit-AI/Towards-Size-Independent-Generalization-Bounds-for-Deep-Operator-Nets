import jax.numpy as jnp
from jax import random, vmap

if __name__=="__main__":
    P_test = 121
    N_test = 1
    key = random.PRNGKey(1122) # a new unseen key
    keys = random.split(key, N_test)

    config.update("jax_enable_x64", True)
    f_test_vis, z_test_vis, u_test_vis = generate_test_data_visualization(key, P_test)

    # #Reshape Data
    # f_test = jnp.float32(f_test.reshape(N_test * P_test,-1))
    # z_test = jnp.float32(z_test.reshape(N_test * P_test,-1))
    # u_test = jnp.float32(u_test.reshape(N_test * P_test,-1))

    # Predict
    params = model.get_params(model.opt_state)
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