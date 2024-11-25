from jax import random, jit, lax
import jax.numpy as jnp

from functools import partial


def generate_initial_condition(N, A, key):
    """Generate an initial condition that is 2π-periodic and has zero mean in the interval [-π, π].

    Args:
        N (int): Number of sine terms to include in the Fourier series.
        A (float): variance of the Gaussian from which the coefficients are sampled.
        key (jax.random.PRNGKey): A random number generator key.

    Returns:
        callable: A function representing the initial condition.
    """
    # Generate random coefficients for the sine terms
    coefficients = A * random.normal(key, (N,))

    def initial_condition(x):
        sine_terms = jnp.zeros_like(x)
        for n in range(N):
            sine_terms += coefficients[n] * jnp.sin((n + 1) * x)

        # Set small values to zero
        #threshold = 1e-14
        #sine_terms = jnp.where(jnp.abs(sine_terms) < threshold, 0, sine_terms)

        return sine_terms

    return initial_condition

# A numerical solver for Burgers' equation
def solve_burgers(key, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period, kappa, m):
    """Solve the 1D Burgers' equation u_t + uu_x = k * u_xx with a given initial condition derived from
    the Fourier Series sine representation and periodic boundary conditions.
    Also generate input and output sensor locations and measurements."""
    xmin, xmax = -period*jnp.pi, period*jnp.pi
    tmin, tmax = 0, T_lim

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Generate the initial condition function
    initial_condition_fn = generate_initial_condition(num_sine_terms, sine_amplitude, subkeys[0])

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute initial condition
    u0 = initial_condition_fn(x)

    # Finite Difference Approximation Matrices
    D1 = jnp.eye(Nx, k=1) - jnp.eye(Nx, k=-1) # first derivative approximation matrix
    D2 = -2 * jnp.eye(Nx) + jnp.eye(Nx, k=-1) + jnp.eye(Nx, k=1)
    D3 = jnp.eye(Nx - 2) # enforce BCs
    M = -jnp.diag(D1 @ (kappa * jnp.ones_like(x))) @ D1 - 4 * jnp.diag(kappa * jnp.ones_like(x)) @ D2
    m_bond = 8 * h ** 2 / dt * D3 + M[1:-1, 1:-1]
    c = 8 * h ** 2 / dt * D3 - M[1:-1, 1:-1]

    u = jnp.zeros((Nx, Nt))
    u = u.at[:, 0].set(u0)

    def body_fn(i, u):
        u_x = D1 @ u[:, i]
        nonlinear_term = u[1:-1, i] * u_x[1:-1]
        b2 = c @ u[1:-1, i].T - nonlinear_term * h ** 2 / 2
        u = u.at[1:-1, i + 1].set(jnp.linalg.solve(m_bond, b2))
        return u

    s = lax.fori_loop(0, Nt - 1, body_fn, u) # PDE solution over Nx x Nt grid

    # Input sensor locations and measurements
    xx = jnp.linspace(xmin, xmax, m)
    u = initial_condition_fn(xx)

    return (x, t, s), (u, u0)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, num_sine_terms, sine_amplitude, Nx , Nt, T_lim, period, kappa):
    # Numerical solution
    (x, t, s), (u, u0) = solve_burgers(key, num_sine_terms, sine_amplitude, Nx , Nt, T_lim, period, kappa)

    # u is 1 x m

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Sampled input data
    u_train = jnp.tile(u, (P,1)) # add dimensions-> copy u P times


    # Sample general evaluation points for DeepONet
    x_2_idx = random.choice(subkeys[0], jnp.arange(Nx), shape = (P,1))
    x_2 = x[x_2_idx]

    t_2_idx = random.choice(subkeys[1], jnp.arange(Nt), shape = (P,1))
    t_2 = t[t_2_idx]

    y_train_don = jnp.hstack((x_2, t_2))


    s_train_don = s[x_2_idx, t_2_idx]

    return u_train, y_train_don, s_train_don

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, P, num_sine_terms, sine_amplitude, T_lim, period, kappa):
    Nx = P
    Nt = P
    (x, t, s), (u, u0) = solve_burgers(key, num_sine_terms, sine_amplitude, Nx , Nt, T_lim, period, kappa)

    XX, TT = jnp.meshgrid(x, t)

    u_test = jnp.tile(u, (P**2,1))
    y_test = jnp.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = s.T.flatten()

    return u_test, y_test, s_test

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