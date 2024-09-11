from jax import random, jit
import jax.numpy as jnp

import matplotlib.pyplot as plt
from functools import partial


def generate_fourier_sine(N, A, x0, key):
    """Generate an Fourier Series representation sine function that's 2π-periodic and has zero mean in the interval [-π, π].

    Args:
        N (int): Number of sine terms to include in the Fourier series.
        A (float): Amplitude of sine terms.
        key (jax.random.PRNGKey): A random number generator key.

    Returns:
        callable: A Fourier-sine function.
    """
    # Generate random coefficients for the sine terms
    # coefficients = A * random.normal(key, (N,))
    coefficients = random.uniform(key, shape=(N,), minval=100, maxval=200)

    def sine_function(x):
        sine_terms = jnp.zeros_like(x)
        for m in range(1,N+1):
            sine_terms += coefficients[m-1] * jnp.sin(m * x / x0)

        # Set small values to zero
        threshold = 1e-14
        sine_terms = jnp.where(jnp.abs(sine_terms) < threshold, 0, sine_terms)

        return sine_terms

    return sine_function, coefficients

def generate_f(x, y, N, A, x0, y0, key):
    # Generate subkeys
    subkeys = random.split(key, 2)
    f1, _ = generate_fourier_sine(N, A, x0, subkeys[0])
    f2, _ = generate_fourier_sine(N, A, y0, subkeys[1])

    return jnp.outer(f1(x), f2(y)).reshape(-1,1)

# 3D plot function
def plot_3d(ax, X, Y, u):
    surf = ax.plot_surface(X, Y, u, cmap='plasma')
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

def f_testing(N, key, x0, y0, sine_amplitude):
    subkeys = random.split(key, 2)
    f1, c_m = generate_fourier_sine(N, sine_amplitude, x0, subkeys[0])
    f2, d_n = generate_fourier_sine(N, sine_amplitude, y0, subkeys[1])
    f_xy = lambda x,y : jnp.outer(f1(x), f2(y)).reshape(-1,1)
    return f_xy, c_m, d_n
  
def v_mn(m, n, x, y, x0, y0): 
    threshold = 1e-14
    sine_terms = jnp.sin(m * jnp.pi * x / x0) * jnp.sin(n * jnp.pi * y / y0)
    return jnp.where(jnp.abs(sine_terms) < threshold, 0, sine_terms)

def A_mn(c_m, d_n):
  return jnp.outer(c_m, d_n)

def u(A_mn, x, y, t, x0, y0):
    u_total = 0
    for m in range(1, A_mn.shape[0]+1):
      for n in range(1, A_mn.shape[1]+1):
        u_total +=  A_mn[m-1][n-1] * v_mn(m, n, x, y, x0, y0) * jnp.exp(-(jnp.pi**2)*((m**2/x0**2)+(n**2/y0**2)) * t)
    return u_total

# Geneate train data corresponding to one input sample
def generate_one_training_data(key, P_train, x0, y0, T_lim, m, sine_amplitude):

    # Generate subkeys
    subkeys = random.split(key, 6)
    # Sample collocation points
    x_train = random.uniform(subkeys[0], minval = 0, maxval = x0, shape = (round(P_train**(1/3)),))
    y_train = random.uniform(subkeys[1], minval = 0, maxval = y0, shape = (round(P_train**(1/3)),))
    t_train = random.uniform(subkeys[2], minval = 0, maxval = T_lim + 1e-10, shape = (round(P_train**(1/3)),))
    x_train_mesh, y_train_mesh, t_train_mesh = jnp.meshgrid(x_train, y_train, t_train, indexing="ij")
    x_train = x_train_mesh.flatten().reshape(-1,1)
    y_train = y_train_mesh.flatten().reshape(-1,1)
    t_train = t_train_mesh.flatten().reshape(-1,1)

    # training collocation points
    z_train = jnp.hstack([x_train, y_train, t_train]) # stack coordinates pairwise (x, y, t)

    # Input sensor locations and measurements
    x_sensor = jnp.linspace(0, x0, int(m**0.5), endpoint = True)
    y_sensor = jnp.linspace(0, y0, int(m**0.5), endpoint = True)

    # x_sensor_mesh, y_sensor_mesh = jnp.meshgrid(x_sensor, y_sensor)
    # x_sensor = x_sensor_mesh.flatten().reshape(-1,1)
    # y_sensor = y_sensor_mesh.flatten().reshape(-1,1)

    f, c_m, d_n = f_testing(2, subkeys[3], x0, y0, sine_amplitude)
    f_sensor = f(x_sensor,y_sensor)
    f_train = jnp.tile(f_sensor.T, (P_train,1))
    u_train = u(A_mn(c_m, d_n), z_train[:,0],z_train[:,1],z_train[:,2],x0, y0)

    return f_train, z_train, u_train

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, P_test, x0, y0, T_lim, m, sine_amplitude):

    # Generate subkeys
    subkeys = random.split(key, 6)
    # Sample collocation points
    x_test = random.uniform(subkeys[0], minval = 0, maxval = x0, shape = (round(P_test**(1/3)),))
    y_test = random.uniform(subkeys[1], minval = 0, maxval = y0, shape = (round(P_test**(1/3)),))
    t_test = random.uniform(subkeys[2], minval = 0, maxval = T_lim, shape = (round(P_test**(1/3)),))
    x_test_mesh, y_test_mesh, t_test_mesh = jnp.meshgrid(x_test, y_test, t_test, indexing="ij")
    x_test = x_test_mesh.flatten().reshape(-1,1)
    y_test = y_test_mesh.flatten().reshape(-1,1)
    t_test = t_test_mesh.flatten().reshape(-1,1)

    # Testing collocation points
    z_test = jnp.hstack([x_test, y_test, t_test]) # stack coordinates pairwise (x, y, t)

    # Input sensor locations and measurements
    x_sensor = jnp.linspace(0, x0, int(m**0.5), endpoint = True)
    y_sensor = jnp.linspace(0, y0, int(m**0.5), endpoint = True)

    # x_sensor_mesh, y_sensor_mesh = jnp.meshgrid(x_sensor, y_sensor)
    # x_sensor = x_sensor_mesh.flatten().reshape(-1,1)
    # y_sensor = y_sensor_mesh.flatten().reshape(-1,1)

    f, c_m, d_n = f_testing(2, subkeys[3], x0, y0, sine_amplitude)
    f_sensor = f(x_sensor,y_sensor)
    f_test = jnp.tile(f_sensor.T, (P_test,1))
    u_test = u(A_mn(c_m, d_n), z_test[:,0],z_test[:,1],z_test[:,2],x0, y0)

    return f_test, z_test, u_test

# Dataset generator class
class DataGenerator:
    def __init__(self, f, z, labels, batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.f = f  # input
        self.z = z  # location
        self.labels = labels  # labeled data evaluated at y

        self.N = f.shape[0]
        self.batch_size = batch_size
        self.key = rng_key
        self.index = 0  # Iterator state

    def __iter__(self):
        'Return the iterator object itself'
        self.index = 0  # Reset the iterator
        return self

    def __next__(self):
        'Generate the next batch of data'
        if self.index >= self.N:
            raise StopIteration  # Stop when all data has been processed

        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        self.index += self.batch_size  # Move the index forward
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        f = self.f[idx, :]
        z = self.z[idx, :]
        labels = self.labels[idx, :]
        # Construct batch
        inputs = (f, z)
        outputs = labels
        return inputs, outputs