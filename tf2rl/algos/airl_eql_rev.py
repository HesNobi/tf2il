import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense

# from tf2rl.networks.layer import EqlLayer, DenseLayer
from tf2rl.networks.layer import DenseLayer, EqlLayerv2 as EqlLayer
from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.algos.gail import Discriminator


class StateActionModel(Discriminator):

    def compute_reward(self, inputs):
        raise NotImplemented("compute_reward of airl_eql_rev is not implemented!")
        # print("[DEBUG] initializing {compute_reward StateActionModel}")
        # if not self._enable_sn:
        #     return tf.math.log(self(inputs) + 1e-8)
        #     # return self(inputs)
        # else:
        #     return tf.math.log(self(inputs, training=False) + 1e-8)
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
            raise NotImplemented
            # self.l1 = tfa.layers.SpectralNormalization(Dense(units[0], name="L1", activation=tf.nn.tanh))
            # self.l2 = tfa.layers.SpectralNormalization(Dense(units[1], name="L2", activation=tf.nn.tanh))
            # self.l3 = tfa.layers.SpectralNormalization(Dense(1, name="L3", activation=output_activation))

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

class StateModel_EQL(tf.keras.Model):
    def __init__(self, state_shape, num_layers=2, drop_out = None,
                 enable_sn=False, output_activation=None, lmbda=0,
                 name="Discriminator", v=None, exclude=None, atol=0, constraint=None,
                 w_initializer='random_normal', b_initializer='random_normal',
                 is_lmbda_dynamic = False):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        # self.w_initializer = initializers.Constant(value=0.1)
        # self.b_initializer = initializers.Constant(value=0.1)
        self.atol= atol
        self.lmbda = lmbda
        self.drop_out = drop_out
        self._is_lmbda_dynamic = is_lmbda_dynamic
        self.output_activation = output_activation

        if drop_out is not None:
            assert (np.array(drop_out) <= 1).all() and (np.array(drop_out) >= 0).all(), "drop_out is a ratio and it needs to be between 0 and 1"
            if isinstance(drop_out, list):
                assert len(drop_out) == num_layers, "Lenth of dropout dosent match the number of the layers."

        if exclude:
            assert len(exclude) == self.num_layers, "exclude parameter wrong format, len(exclude) must equal " \
                                                    "num_layers, ex.: exclude=[['sin'],[]], num_layers = 2"
            self.exclude = exclude
        else:
            exclude = [[] for i in range(self.num_layers)]
            self.exclude = exclude

        if v is None:
            self.v = np.ones(self.num_layers, dtype=int)
        else:
            self.v = v

        assert len(self.v) == self.num_layers, 'v array must have same dimensions as number of hidden layers param'

        self.eql_layers = []
        # self.eql_layers.append(tfa.layers.SpectralNormalization(Dense(state_shape[0]+action_dim,
        #                                                          name="spec_layer",activation=tf.tanh,
        #                                                          kernel_initializer=initializers.glorot_uniform())))
        for index in range(self.num_layers):
            self.eql_layers.append(EqlLayer(w_initializer=self.w_initializer, b_initializer=self.b_initializer,
                                            v=self.v[index], exclude=self.exclude[index],
                                            constraint=constraint, lmbda=lmbda, dynamic_lmbda=self._is_lmbda_dynamic))

            if drop_out and not isinstance(drop_out, list):
                self.eql_layers.append(tf.keras.layers.Dropout(drop_out))
            elif drop_out and isinstance(drop_out, list):
                if drop_out[index] > 0:
                    self.eql_layers.append(tf.keras.layers.Dropout(drop_out[index]))

        self.layer_out = DenseLayer(w_initializer=self.w_initializer, b_initializer=self.b_initializer,
                                    constraint=constraint, lmbda=lmbda, dynamic_lmbda=self._is_lmbda_dynamic,
                                    activation=output_activation)

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        with tf.device("/cpu:0"):
            input_tf = tf.concat((dummy_state, dummy_action), axis=1)
            print("[DEBUG] initializing dummy variable. [__init__ StateModel_EQL] input_shape: {} output_shape: {} name: {}"
                  .format(input_tf.shape, self(input_tf, training=False).shape, self.name))

    def call(self, inputs, training=None, l1_regularizers = 0):
        print("[DEBUG] initializing [call Discriminator_EQL shapes: {} name: {}".format(inputs.shape, self.name))

        if training is None:
            training = True
        features = tf.cast(inputs, tf.float32)
        for layer in self.eql_layers:
            if "dropout" in layer.name or "spec" in layer.name:
                features = layer(features,
                                 training=training)
            else:
                features = layer(features,
                                 training=training,
                                 l1_regularizers=l1_regularizers)

        return self.layer_out(features, training=training, l1_regularizers=l1_regularizers)

    def compute_reward(self, inputs):
        raise NotImplemented("compute_reward of airl_eql_rev is not implemented!")

class AIRL_EQL_REV(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim,
            lr=0.001,
            num_layers=2,
            name="AIRL_EQL_REV",
            enable_sn=False,
            is_debug=False,
            is_state_only=False,
            update_delay=1,
            atol=0,
            atol_starting_step=0,
            v=[1, 1],
            units = [64,64],
            mask_rate=0,
            masking_starting_step=0,
            lmbda=0,
            regularizer_starting_step=0,
            drop_out=None,
            tau=None,
            exclude=None,
            **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)
        self._state_only = is_state_only
        self._is_debug = is_debug
        self.update_delay = update_delay
        self.atol = atol
        self.atol_starting_step = atol_starting_step if atol > 0 else None
        self.v = v
        self.mask_rate = mask_rate
        self.masking_starting_step = masking_starting_step if mask_rate > 0 else None
        self.lmbda = lmbda
        self.regularizer_starting_step = regularizer_starting_step if lmbda > 0 else None
        self._is_reg_dyna = False  # True if self.regularizer_starting_step is not None else False
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.enable_sn = enable_sn
        self.drop_out = drop_out if drop_out is not None else None
        self.tau = tau
        self.lr = lr
        self.exclude = exclude

        if self._state_only:
            self.rew_net = StateModel(
                state_shape=state_shape, units=units,
                name="reward_net", enable_sn=enable_sn, output_activation="linear")
        else:
            self.rew_net = StateActionModel(
                state_shape=state_shape, action_dim=action_dim, units=units,
                name="reward_net", enable_sn=enable_sn, output_activation="linear")

        self.val_net = StateModel_EQL(
            state_shape=self.state_shape, v=self.v, drop_out=self.drop_out,
            num_layers=self.num_layers, enable_sn=self.enable_sn, constraint=None,
            name="val_eql_net", lmbda=self.lmbda, exclude=self.exclude,
            is_lmbda_dynamic=self._is_reg_dyna, output_activation=None)

        self.rew_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def train(self, agent_states, agent_acts, agent_next_states, agent_logps,
              expert_states, expert_acts, expert_next_states, expert_logps, itr):

        if itr % self.update_delay == 0:
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
        print("[DEBUG] initializing {_train_body AIRL_EQL}.")
        with tf.device(self.device):
            with tf.GradientTape() as tape:  # persistent=True
                tape.watch(self.rew_net.trainable_variables +
                           self.val_net.trainable_variables)
                if self._state_only:
                    real_rews = self.rew_net(expert_states)
                    fake_rews = self.rew_net(agent_states)
                else:
                    real_rews = self.rew_net(tf.concat([expert_states, expert_acts], axis=1))
                    fake_rews = self.rew_net(tf.concat([agent_states, agent_acts], axis=1))
                real_vals = self.val_net(tf.concat([expert_states, expert_next_states], axis=1))
                fake_vals = self.val_net(tf.concat([agent_states, agent_next_states], axis=1))

                log_p_tau_real = real_rews + self.discount * real_vals
                log_p_tau_fake = fake_rews + self.discount * fake_vals
                # log_p_tau_real = real_rews + real_vals
                # log_p_tau_fake = fake_rews + fake_vals

                log_pq_real = tf.reduce_logsumexp([log_p_tau_real, expert_logps], axis=0)
                log_pq_fake = tf.reduce_logsumexp([log_p_tau_fake, agent_logps], axis=0)

                loss = -(tf.reduce_mean(log_p_tau_real - log_pq_real) +
                         tf.reduce_mean(agent_logps - log_pq_fake) -
                         1 * tf.reduce_sum(self.val_net.losses))

            grads = tape.gradient(loss, self.rew_net.trainable_variables + self.val_net.trainable_variables)
            self.rew_optimizer.apply_gradients(
                zip(grads, self.rew_net.trainable_variables + self.val_net.trainable_variables))

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
        print("[DEBUG] initializing {_inference_body AIRL}.")
        with tf.device(self.device):
            if self._state_only:
                return self.rew_net(states, training=False) + \
                       self.discount * self.val_net(tf.concat([states, next_states], axis=1),
                                                   training=False) # - \
                      # logps
            else:
                return self.rew_net(tf.concat([states, actions], axis=1), training=False) + \
                       self.discount * self.val_net(tf.concat([states, next_states], axis=1),
                                                    training=False)# - \
                       # logps

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
        # over riding --units-ail to be [64,64]
        parser.add_argument('--units-ail', type=str, default="64,64",
                            help='Discriminator architecture. use case: 128,128 or 400,300 <default: %(default)s>')
        parser.add_argument('--update-delay', type=int, default=3,
                            help='Discriminator training delay.  <default: %(default)s>')
        # parser.add_argument('--dropout', type=float, default=0.5,
        #                     help='Discriminator dropout.  <default: %(default)s>')
        parser.add_argument('--dropout', type=str, default='.5,0.',
                            help='Discriminator dropout. use case: .75,5 or 0.5  <default: %(default)s>')
        parser.add_argument('-v', '--v', type=str, default="2,1",
                            help='Discriminator architecture. use case: 2,1 or 1. <default: %(default)s>')
        parser.add_argument('--exclude', type=str, default=None,
                            help='Discriminator architecture. use case: None or sig,sig or cos.  <default: %(default)s>')
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
