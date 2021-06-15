from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her_sac.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.her_sac.normalizer import Normalizer
from baselines.her_sac.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


global DEMO_BUFFER #buffer for demonstrations

class SAC(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 sample_transitions, gamma, alpha, double_Q_trick, reuse=False, **kwargs):
        """Implementation of SAC that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her_sac.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per SAC agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g', 'u']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        global DEMO_BUFFER
        DEMO_BUFFER = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions) #initialize the demo buffer; in the same way as the primary data buffer

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_u(self, o, g):
        """Sample an a' with pi_theta """
        policy = self.main1
        # values to compute
        vals = [policy.pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        u = self.sess.run(vals, feed_dict=feed)[0]
        # action postprocessing
        # noise = np.clip(self.policy_noise*np.random.randn(*u.shape), -self.noise_clip, self.noise_clip)  # gaussian noise
        # u = np.clip(u + noise, -self.max_u, self.max_u)
        return u

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None


    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target1 if use_target_net else self.main1
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        # No need to noise anymore since policy is stochastic
        # noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        # u += noise
        # u = np.clip(u, -self.max_u, self.max_u)
        # u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            episode_batch['u_2'] = episode_batch['u'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q1_adam.sync()
        self.Q2_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic1_loss, critic2_loss, actor_loss, Q1_grad, Q2_grad, pi_grad = self.sess.run([
            self.Q1_loss_tf,
            self.Q2_loss_tf,
            self.main1.Q_pi_tf,
            self.Q1_grad_tf,
            self.Q2_grad_tf,
            self.pi_grad_tf
        ])
        return critic1_loss, critic2_loss, actor_loss, Q1_grad, Q2_grad, pi_grad

    def _update(self, Q1_grad, Q2_grad, pi_grad):
        self.Q1_adam.update(Q1_grad, self.Q_lr)
        self.Q2_adam.update(Q2_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        if self.bc_loss: #use demonstration buffer to sample as well if bc_loss flag is set TRUE
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            global DEMO_BUFFER
            transitions_demo = DEMO_BUFFER.sample(self.demo_batch_size) #sample from the demo buffer
            for k, values in transitions_demo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
        else:
            transitions = self.buffer.sample(self.batch_size) #otherwise only sample from primary buffer

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        transitions['u_2'] = self._preprocess_u(o_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic1_loss, critic2_loss, actor_loss, Q1_grad, Q2_grad, pi_grad = self._grads()
        self._update(Q1_grad, Q2_grad, pi_grad)
        return critic1_loss, critic2_loss, actor_loss

    def _init_target1_net(self):
        self.sess.run(self.init_target1_net_op)
    
    def _init_target2_net(self):
        self.sess.run(self.init_target2_net_op)

    def update_target1_net(self):
        self.sess.run(self.update_target1_net_op)
    
    def update_target2_net(self):
        self.sess.run(self.update_target2_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a SAC agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main1', reuse=tf.AUTO_REUSE) as vs:
            if reuse:
                vs.reuse_variables()
            self.main1 = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            main1_next_batch_tf = batch_tf.copy()
            main1_next_batch_tf['o'] = batch_tf['o_2']
            main1_next_batch_tf['g'] = batch_tf['g_2']
            main1_next_batch_tf['u'] = batch_tf['u_2']
            self.main1_next = self.create_actor_critic(main1_next_batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()

        with tf.variable_scope('target1') as vs:
            if reuse:
                vs.reuse_variables()
            target1_batch_tf = batch_tf.copy()
            target1_batch_tf['o'] = batch_tf['o_2']
            target1_batch_tf['g'] = batch_tf['g_2']
            target1_batch_tf['u'] = batch_tf['u_2']
            self.target1 = self.create_actor_critic(
                target1_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main1")) == len(self._vars("target1"))
        
        with tf.variable_scope('main2') as vs:
            if reuse:
                vs.reuse_variables()
            self.main2 = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target2') as vs:
            if reuse:
                vs.reuse_variables()
            target2_batch_tf = batch_tf.copy()
            target2_batch_tf['o'] = batch_tf['o_2']
            target2_batch_tf['g'] = batch_tf['g_2']
            target2_batch_tf['u'] = batch_tf['u_2']
            self.target2 = self.create_actor_critic(
                target2_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main2")) == len(self._vars("target2"))

        # loss functions
        target1_Q_tf = self.target1.Q_tf
        target2_Q_tf = self.target2.Q_tf
        log_pi_u_tf = self.main1_next.log_pi_u_tf # we use main1_next because the state for the log is s'
        
        if self.double_Q_trick:
            target_Q_pi_tf = tf.minimum(target1_Q_tf, target2_Q_tf) - self.alpha*log_pi_u_tf # cata when alpha!=0
        else:
            target_Q_pi_tf = target1_Q_tf - self.alpha*log_pi_u_tf
        
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q1_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main1.Q_tf))
        self.Q2_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main2.Q_tf))

        main1_Q_tf = self.main1.Q_pi_tf
        main2_Q_tf = self.main2.Q_pi_tf
        log_pi_tf = self.main1.log_pi_tf
        if self.double_Q_trick:
            self.target_pi = tf.minimum(main1_Q_tf, main2_Q_tf) - self.alpha * log_pi_tf
        else:
            self.target_pi = main1_Q_tf - self.alpha * log_pi_tf
        
        self.pi_loss_tf = -tf.reduce_mean(self.target_pi)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main1.pi_tf / self.max_u))

        Q1_grads_tf = tf.gradients(self.Q1_loss_tf, self._vars('main1/Q'))
        Q2_grads_tf = tf.gradients(self.Q2_loss_tf, self._vars('main2/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main1/pi'))
        assert len(self._vars('main1/Q')) == len(Q1_grads_tf)
        assert len(self._vars('main2/Q')) == len(Q2_grads_tf)
        assert len(self._vars('main1/pi')) == len(pi_grads_tf)
        # self.Q1_grads_vars_tf = zip(Q1_grads_tf, self._vars('main1/Q'))
        # self.Q2_grads_vars_tf = zip(Q2_grads_tf, self._vars('main2/Q'))
        # self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main1/pi'))
        self.Q1_grad_tf = flatten_grads(grads=Q1_grads_tf, var_list=self._vars('main1/Q'))
        self.Q2_grad_tf = flatten_grads(grads=Q2_grads_tf, var_list=self._vars('main2/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main1/pi'))

        # optimizers
        self.Q1_adam = MpiAdam(self._vars('main1/Q'), scale_grad_by_procs=False)
        self.Q2_adam = MpiAdam(self._vars('main2/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main1/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.pi_vars = self._vars('main1/pi') # pi_1

        self.main_Q1_vars = self._vars('main1/Q')  # Q_1
        self.main_Q2_vars = self._vars('main2/Q')  # Q_2

        self.target1_vars = self._vars('target1/Q')    # Q_targ_1
        self.target2_vars = self._vars('target2/Q')    # Q_targ_2

        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target1_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target1_vars, self.main_Q1_vars)))
        self.init_target2_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target2_vars, self.main_Q2_vars)))

        self.update_target1_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target1_vars, self.main_Q1_vars)))
        self.update_target2_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target2_vars, self.main_Q2_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target1_net()
        self._init_target2_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main1', 'target1', 'target2', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)

