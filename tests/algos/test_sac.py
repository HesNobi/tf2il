import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.sac import SAC
from tests.algos.common import CommonContinuousOutputAlgos

config = tf.ConfigProto(allow_soft_placement=True)
tf.enable_eager_execution(config=config)


class TestSAC(CommonContinuousOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SAC(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
