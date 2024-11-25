import jax.numpy as jnp
from jax.nn import relu, silu
from jax import random, grad, vmap, jit
import optax

from tqdm import trange
from functools import partial


# Define the neural net
def MLP(layers, activation):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
        #   b = jnp.zeros(d_out)
          return W#, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
    #   for W, b in params[:-1]:
      for W in params[:-1]:
          outputs = jnp.dot(inputs, W) #+ b
          inputs = activation(outputs)
    #   W, b = params[-1]
      W = params[-1]
      outputs = jnp.dot(inputs, W) #+ b
      return outputs
  return init, apply

# Define the model
class DeepONet:
    def __init__(self, branch_layers, trunk_layers, loss_type="l2", huber_delta=0.4, activation=relu):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=activation)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=activation)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        self.params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer = optax.adam(optax.exponential_decay(1e-3, transition_steps=2000,decay_rate=0.9))
        self.opt_state = self.optimizer.init(self.params)

        # Loss Setup
        self.loss_type = loss_type # huber or l2
        self.huber_delta = huber_delta

        # Loggers
        self.loss_don_log = []
        self.loss_test_log = []
        self.loss_AF_test_log = []

    # Define DeepONet architecture
    def operator_net(self, params, f, x, y, t):
        branch_params, trunk_params = params
        z = jnp.hstack([x.reshape(-1,1), y.reshape(-1,1), t.reshape(-1,1)])
        B = self.branch_apply(branch_params, f)
        T = self.trunk_apply(trunk_params, z)
        outputs = jnp.sum(B * T)
        return  outputs

    # Define DeepONet loss
    def loss_don(self, params, batch):
        inputs, outputs = batch
        f, z = inputs
        # Compute forward pass
        pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, f, z[:,0], z[:,1], z[:,2])
        # Compute loss
        if(self.loss_type == "huber"):
            huber_elem = optax.losses.huber_loss(pred, outputs.flatten(), delta = self.huber_delta)
            loss = jnp.mean(huber_elem)
        elif(self.loss_type == "l2"):
            loss = jnp.mean((outputs.flatten() - pred)**2)
        return loss

    # Define Test loss
    def loss_test(self, params, f_test, z_test, u_test):
        u_pred = self.predict_u(params, f_test, z_test)
        if(self.loss_type == "huber"):
            huber_elem = optax.losses.huber_loss(u_pred,u_test,delta = self.huber_delta)
            return jnp.mean(huber_elem)
        elif(self.loss_type == "l2"):
            return jnp.mean((u_test - u_pred)**2)

    # Define Average Fractional Test loss
    def AF_loss_test(self, params, f_test, z_test, u_test):
      u_pred = self.predict_u(params, f_test, z_test)
      if(self.loss_type == "huber"):
          huber_elem = optax.losses.huber_loss(u_pred,u_test,delta = self.huber_delta)
          u_mean = jnp.mean(huber_elem)
          return u_mean / jnp.mean((u_test)**2)
      elif(self.loss_type == "l2"):
          return jnp.mean((u_test - u_pred)**2) / jnp.mean((u_test)**2)

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, don_batch):
        g = grad(self.loss_don)(params, don_batch)
        updates, opt_state = self.optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Optimize parameters in a loop
    def train(self, don_dataset, test_dataset, nIter):
        # Define data iterators
        f_test, z_test, u_test = test_dataset

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Process the batch
            for don_batch in don_dataset:
                self.params, self.opt_state = self.step(self.params, self.opt_state, don_batch)
            
            if it % 100 == 0:
                # Compute losses
                loss_don_value = self.loss_don(self.params, don_batch)
                loss_test_value = self.loss_test(self.params, f_test, z_test, u_test)
                AFTL_value = self.AF_loss_test(self.params, f_test, z_test, u_test)

                # Store losses
                self.loss_don_log.append(loss_don_value)
                self.loss_test_log.append(loss_test_value)
                self.loss_AF_test_log.append(AFTL_value)

                # Print losses
                pbar.set_postfix({'Training Loss': loss_don_value,'Test Loss': loss_test_value, 'AFTL': AFTL_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, f_star, z_star):
        u_pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, f_star, z_star[:,0], z_star[:,1], z_star[:,2])
        return u_pred.reshape(-1,1)