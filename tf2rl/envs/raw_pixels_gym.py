import numpy as np
import collections

from gym import spaces
from gym import ObservationWrapper
from gym.wrappers.pixel_observation import *
from tf2rl.envs.frame_stack_wrapper import FrameStack

class Obs2RawPixel(ObservationWrapper):
    def __init__(self, env, is_gray=True):
        super(Obs2RawPixel, self).__init__(env)
        assert isinstance(env, PixelObservationWrapper) or isinstance(env, PixelObservationWrapper_classic), \
            "This Wrapper should be used after gym.wrappers.pixel_observation.PixelObservationWrapper or _classic"

        # self.env  = env
        self._is_gray = is_gray

        n_channel = 1 if is_gray else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=tuple(list(env.observation_space.shape[:-1]) + [n_channel]),
                                            dtype=np.uint8)
        self._max_episode_steps = env._max_episode_steps

    def observation(self, img):
        # img = observation
        img = img[:,:,::-1]
        if self._is_gray:
            # img = img.astype(np.float32)
            img = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114)
            img = np.expand_dims(img, axis=-1).astype(np.uint8)
        assert img.dtype == np.uint8, "The Observations data type should be np.uint8!"
        return img

def ControlPixelWrapper(env, n_stack=1, width=48, height=48, is_gray=True):
    env = PixelObservationWrapper(env, render_shape=(width,height))
    # env = PixelObservationWrapper_classic(env, render_shape=(width,height))
    env = Obs2RawPixel(env,  is_gray=is_gray)
    if n_stack > 1:
        env = FrameStack(env, n_stack)
    return env