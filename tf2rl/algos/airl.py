import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.algos.gail import Discriminator #as StateActionModel
# from tf2rl.networks.spectral_norm_dense import SNDense
import tensorflow_addons as tfa

class StateActionModel(Discriminator):

    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward StateActionModel}")
        if not self._enable_sn:
            return tf.math.log(self(inputs) + 1e-8)
            # return self(inputs)
        else:
            return tf.math.log(self(inputs, training=False) + 1e-8)
            # return self(inputs, training=False)

class StateModel(tf.keras.Model):
    def __init__(self, state_shape, units=[32, 32],
                 enable_sn=False, output_activation="sigmoid",
                 name="StateModel"):
        super().__init__(name=name)
        self._enable_sn = enable_sn

        if not enable_sn:
            self.l1 = Dense(units[0], name="L1", activation=tf.nn.tanh)
            self.l2 = Dense(units[1], name="L2", activation=tf.nn.tanh)
            self.l3 = Dense(1, name="L3", activation=output_activation)
        else:
            print("[DEBUG] Spectral Normalization is enabled")
            self.l1 = tfa.layers.SpectralNormalization(Dense(units[0], name="L1", activation=tf.nn.tanh))
            self.l2 = tfa.layers.SpectralNormalization(Dense(units[1], name="L2", activation=tf.nn.tanh))
            self.l3 = tfa.layers.SpectralNormalization(Dense(1, name="L3", activation=output_activation))

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state)

    def call(self, inputs, training=None):
        if training == None:
            training = True

        if not self._enable_sn:
            features = self.l1(inputs)
            features = self.l2(features)
            return self.l3(features)
        else:
            features = self.l1(inputs, training=training)
            features = self.l2(features, training=training)
            return self.l3(features, training=training)


class AIRL(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim,
            state_only=True,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            name="AIRL",
            is_debug = False,
            **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)
        self._state_only = state_only
        self._is_debug = is_debug
        if state_only:
            self.rew_net = StateModel(
                state_shape=state_shape, units=units,
                name="reward_net", enable_sn=enable_sn, output_activation="linear")
        else:
            self.rew_net = StateActionModel(
                state_shape=state_shape, action_dim=action_dim, units=units,
                name="reward_net", enable_sn=enable_sn, output_activation="linear")
        self.val_net = StateModel(
            state_shape=state_shape, units=units, enable_sn=enable_sn,
            name="val_net", output_activation="linear")
        self.rew_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # self.val_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def train(self, agent_states, agent_acts, agent_next_states, agent_logps,
              expert_states, expert_acts, expert_next_states, expert_logps, itr):

        loss, rewf, valf = self._train_body(
            agent_states, agent_acts, agent_next_states, agent_logps,
            expert_states, expert_acts, expert_next_states, expert_logps)
        tf.summary.scalar(name=self.policy_name+"/Loss", data=loss)
        tf.summary.scalar(name=self.policy_name + "/rewardF", data=rewf)
        tf.summary.scalar(name=self.policy_name + "/valueF", data=valf)
        if self._is_debug:
            [tf.summary.histogram(name=self.policy_name + "/reward: {}".format(i.shape), data=i) for i in
             self.rew_net.get_weights()]
            [tf.summary.histogram(name=self.policy_name + "/val: {}".format(i.shape), data=i) for i in
             self.val_net.get_weights()]

    @tf.function
    def _train_body(self, agent_states, agent_acts, agent_next_states, agent_logps,
                    expert_states, expert_acts, expert_next_states, expert_logps):
        print("[DEBUG] initializing {_train_body AIRL}.")
        with tf.device(self.device):
            with tf.GradientTape() as tape: # persistent=True
                tape.watch(self.rew_net.trainable_variables+
                           self.val_net.trainable_variables)
                if self._state_only:
                    real_rews = self.rew_net(expert_states)
                    fake_rews = self.rew_net(agent_states)
                else:
                    real_rews = self.rew_net(tf.concat([expert_states, expert_acts], axis=1))
                    fake_rews = self.rew_net(tf.concat([agent_states, agent_acts], axis=1))
                real_vals = self.val_net(expert_states)
                real_next_vals = self.val_net(expert_next_states)
                fake_vals = self.val_net(agent_states)
                fake_next_vals = self.val_net(agent_next_states)

                # # TF2RL original lost.
                # loss = tf.reduce_mean(
                #     fake_rews + self.discount * fake_next_vals - fake_vals - agent_logps)
                # loss -= tf.reduce_mean(
                #     real_rews + self.discount * real_next_vals - real_vals - expert_logps)

                log_p_tau_real = real_rews + self.discount * real_next_vals - real_vals
                log_p_tau_fake = fake_rews + self.discount * fake_next_vals - fake_vals
                log_pq_real = tf.reduce_logsumexp([log_p_tau_real, expert_logps], axis=0)
                log_pq_fake = tf.reduce_logsumexp([log_p_tau_fake, agent_logps], axis=0)
                loss =  -(tf.reduce_mean(log_p_tau_real - log_pq_real) +
                         tf.reduce_mean(agent_logps - log_pq_fake))

                # Just for test i am switing the expert with agent <--------------------------------------
                # loss = -(tf.reduce_mean(log_p_tau_fake - log_pq_fake) +
                #          tf.reduce_mean(expert_logps - log_pq_real))

            # grads_val = tape.gradient(loss, self.val_net.trainable_variables)
            # grads_rew = tape.gradient(loss, self.rew_net.trainable_variables)
            # self.val_optimizer.apply_gradients(
            #     zip(grads_val, self.val_net.trainable_variables))
            # self.rew_optimizer.apply_gradients(
            #     zip(grads_rew, self.rew_net.trainable_variables))

            grads = tape.gradient(loss, self.rew_net.trainable_variables+self.val_net.trainable_variables)
            self.rew_optimizer.apply_gradients(
                zip(grads, self.rew_net.trainable_variables+self.val_net.trainable_variables))

            del tape

        return loss, tf.reduce_mean(fake_rews), tf.reduce_mean(fake_vals)

    def inference(self, states, actions, next_states, logps=None, **kwargs):
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
            if logps:
                logps = np.expand_dims(logps, axis=0)
                logps = np.expand_dims(logps, axis=0)
        return self._inference_body(states, actions)
        # return self._inference_body_disc(states, actions, next_states, logps)

    @tf.function
    def _inference_body(self, states, actions):
        print("[DEBUG] initializing {_inference_body AIRL}.")
        with tf.device(self.device):
            if self._state_only:
                return self.rew_net(states, training=False)
            else:
                return self.rew_net(tf.concat([states, actions], axis=1), training=False)

    @tf.function
    def _inference_body_disc(self, states, actions, next_states, logps):
        print("[DEBUG] initializing {_inference_body_disc AIRL}.")
        with tf.device(self.device):
            if self._state_only:
                return self.rew_net(states, training=False) + \
                      self.discount * self.val_net(next_states, training=False) - \
                      self.val_net(states, training=False) - \
                      logps
            else:
                return self.rew_net(tf.concat([states, actions], axis=1), training=False) + \
                       self.discount * self.val_net(next_states, training=False) - \
                       self.val_net(states, training=False) - \
                       logps

    def inference_bellman(self, states, actions, next_states, logps, **kwargs):
        raise NotImplementedError
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
            if logps:
                logps = np.expand_dims(logps, axis=0)
                logps = np.expand_dims(logps, axis=0)
        return self._inference_bellman_body(states, actions, next_states, logps)

    @tf.function
    def _inference_bellman_body(self, states, actions, next_states, logps):
        print("[DEBUG] initializing {_inference_bellman_body AIRL}.")
        with tf.device(self.device):
            if self._state_only:
                return self.rew_net(states, training=False) + \
                       self.discount * self.val_net(next_states, training=False) - \
                       logps
            else:
                return self.rew_net(tf.concat([states, actions], axis=1), training=False) + \
                       self.discount * self.val_net(next_states, training=False) - \
                       logps

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--state-only', action='store_true',
                            help='Enable the state only AIRL mode. <default: %(default)s>')
        # parser.add_argument('--enable-sn', action='store_true')
        return parser

    # @staticmethod
    # def get_argument(parser=None):
    #     import argparse
    #     if parser is None:
    #         parser = argparse.ArgumentParser(conflict_handler='resolve')
    #     parser.add_argument('--enable-sn', action='store_true')
    #     return parser
