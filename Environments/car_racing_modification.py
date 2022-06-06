from .car_racing import CarRacing
import gym
import supersuit as ss
import matplotlib
import numpy as np

def carRacingGrayscale():
    env = ss.reshape_v0(ss.color_reduction_v0(CarRacing(), 'full'), (96, 96, 1))
    return env

def carRacingGreen():
    env = ss.reshape_v0(ss.color_reduction_v0(CarRacing(), 'G'), (96, 96, 1))
    return env

def carRacingHSVSaturation():
    def obs_converter(obs):
        obs = matplotlib.colors.rgb_to_hsv(obs)[:,:,1]
        obs = np.floor(obs * 255)
        obs = np.expand_dims(obs, axis=2)
        return obs
    env = ss.observation_lambda_v0(CarRacing(), 
                                    lambda obs, obs_space : obs_converter(obs), 
                                    lambda obs_space : gym.spaces.Box(low=0, high=255, shape=(96, 96, 1), dtype=np.uint8)
                                    )
    return env

def carRacingFramestack(num_frames):

    env = ss.frame_stack_v1(ss.color_reduction_v0(CarRacing(), 'full'), num_frames)
    return env

def carRacingResized(size):
    env = CarRacing()
    env = ss.color_reduction_v0(env, 'full')
    env = ss.resize_v0(env, size, size, linear_interp=True)
    env = ss.frame_stack_v1(env, 4)
    return env