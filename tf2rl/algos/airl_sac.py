import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Dense, LayerNormalization

# from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
# from tf2rl.policies.tfp_gaussian_actor import GaussianActor
from tf2rl.algos.sac import SAC
from tf2rl.algos.airl import AIRL

class AIRL_SAC(SAC):
    def __init__(
            self,
            state_shape,
            action_dim,
            # irl,
            name="AIRL_SAC",
            **kwargs):
        super().__init__(state_shape, action_dim, name, **kwargs)
        # self._irl=irl
        # assert isinstance(self._irl, AIRL), "AIRL_SAC needs to have AIRL as its discriminator."

    def train(self, states, actions, next_states, targets, dones, weights=None):

        td_errors, actor_loss, qf1, qf1_loss, logp_min, logp_max, logp_mean = self._train_body(
            states, actions, next_states, targets, dones)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_Q1_value", data=qf1.numpy().mean())
        tf.summary.scalar(name=self.policy_name + "/critic_Q1_loss", data=qf1_loss)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
            tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, targets, dones):
        print("[DEBUG] initializing {_train_body AIRL_SAC}")
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(targets.shape) == 2
            targets = tf.squeeze(targets, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)

                # _, next_logp = self.actor(next_states)
                # sample_next_actions, next_logp = self.actor(next_states)
                # next_q1_target = self.qf1_target(next_states, sample_next_actions)
                # next_q2_target = self.qf2_target(next_states, sample_next_actions)
                # next_q_min_target = tf.minimum(next_q1_target, next_q2_target)
                # soft_next_q = next_q_min_target - self.alpha * next_logp
                # target_q = tf.stop_gradient(rewards + not_dones * self.discount * soft_next_q)
                # TODO: need to think of the end of the episodes
                target_q = tf.stop_gradient(targets)

                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(7)
                td_loss_q = td_loss_q1 + td_loss_q2

                sample_actions, logp = self.actor(states)  # Resample actions to update V
                current_q1_policy = self.qf1(states, sample_actions)
                current_q2_policy = self.qf2(states, sample_actions)
                current_min_q_policy = tf.minimum(current_q1_policy, current_q2_policy)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q_policy)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean((self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            trainable_var = self.qf1.trainable_variables + self.qf2.trainable_variables
            q_grad = tape.gradient(td_loss_q, trainable_var)
            self.qf_optimizer.apply_gradients(zip(q_grad, trainable_var))

            update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=self.tau)

            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

        return target_q, policy_loss, current_q1, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(
            logp)
