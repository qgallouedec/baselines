import tensorflow as tf
from baselines.her_sac.util import store_args, nn

import numpy as np

EPS = 1e-8

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her_sac.Normalizer): normalizer for observations
            g_stats (baselines.her_sac.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            net = tf.nn.relu(nn(input_pi, [self.hidden] * self.layers))
            mu = tf.tanh(tf.layers.dense(net, self.dimu))
            log_std = tf.layers.dense(net, self.dimu)
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)
            pi_tf = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi_tf = gaussian_likelihood(pi_tf, mu, log_std)
            self.mu, self.pi_tf, self.logp_pi_tf = apply_squashing_func(mu, pi_tf, logp_pi_tf)



        with tf.variable_scope('Q'):
            # for policy training (V_pi)
            input_Q = tf.concat(axis=1, values=[o, g, tf.stop_gradient(self.pi_tf) / self.max_u])
            self.Q_pi_tf_sampled = nn(input_Q, [self.hidden] * self.layers + [1])
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

            # for critic training (Q_pi)
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
