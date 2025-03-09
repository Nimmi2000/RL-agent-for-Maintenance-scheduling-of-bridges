import numpy as np
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from enum import IntEnum
import pandas as pd
import datetime
import math

class OBS(IntEnum):
    traf_int = 0
    latitude = 1
    longitude = 2
    age = 3
    aux_age = 4
    fp = 5
    sf = 6
    capital = 7

class Action(IntEnum):
    Nothing = 0
    Monitoring = 1
    Minor_intervention = 2
    Medium_intervention = 3
    Major_intervention = 4
    Replace = 5


class bridgedeterioration(gym.Env):

    def __init__(self, df_path):

        self.current_step = 0
        self.max_step = 200
        self.state = None

        df = pd.read_csv(df_path)
        self.bridge_ids = df['bridge_id'].to_numpy()
        
        bridge_features = df[['traffic_intensity', 'latitude','longitude']].copy()
        bridge_features['age'] = (datetime.datetime.now().year - df['last_replacement']).abs()
        bridge_features['aux_age'] = bridge_features['age'].copy()
        bridge_features['aux_age'] = bridge_features['aux_age'].replace(0, 1) 
        bridge_features['failure_prob'] = 0.0
        bridge_features['reliability'] = 0.0
        bridge_features['capital_invested'] = 0.0

        self.bridges = bridge_features.to_numpy().astype('float32')

        self.num_bridges = len(self.bridges)
        self.observation_shape = self.bridges.shape
        self.n_actions = 3

        self.bridges[:, OBS.sf] = self.calculate_sf(self.bridges, Mu=80, beta = 2)
        self.bridges[:, OBS.fp] = 1 - self.bridges[:, OBS.sf]

        self.action_space = spaces.MultiDiscrete([6] * 15)
        self.observation_space = spaces.Box(0, np.inf , shape=(len(self.bridges), self.bridges.shape[1]), dtype=np.float64)

    def calculate_sf(self, bridges, Mu, beta):
        return np.exp(-(bridges[:, OBS.aux_age]/ (Mu/(1 +(bridges[:, OBS.traf_int]/30000))))**beta)

    def step(self, actions):

        assert len(actions) == self.num_bridges, \
            f"{self.num_bridges} actions are expected, but {len(actions)} were given"
        
        self.current_step += 1
        done = self.current_step >= self.max_step

        if self.state is None:
            raise ValueError("State is None Ensure that it is propesfy initialized before calling calculate_rewards.")
        
        self.state[:, OBS.age] += 1  # Increase actual age of bridge

        # Conditions for 'do nothing' and 'apply maintenance while aux age > 10'
        # If age maintenance is done with aux age <= 10, don't increase aux age

        # conditions = [actions == Action.Nothing, actions == Action.Monitoring, (actions == Action.Minor_intervention) & (self.state[:, OBS.aux_age] > 15), (actions == Action.Medium_intervention) & (self.state[:, OBS.aux_age] > 30), (actions == Action.Major_intervention) & (self.state[:, OBS.aux_age] > 45) ]
        conditions = [actions == Action.Nothing, actions == Action.Monitoring, actions == Action.Minor_intervention, actions == Action.Medium_intervention, actions == Action.Major_intervention]
        choices = [1, 1, -20, -40, -60]
        self.state[:, OBS.aux_age] += np.select(conditions, choices, default= 1)

        self.state[np.where(actions == Action.Replace)[0], OBS.aux_age] = 1  # Set aux age to 1 for replace actions

        # Update failure probabilities
        self.state[:, OBS.sf] = self.calculate_sf(self.state, Mu=80, beta = 2)
        self.state[:, OBS.fp] = 1 - self.state[:, OBS.sf]

        conditions = [actions == Action.Nothing, actions == Action.Monitoring, actions == Action.Minor_intervention, actions == Action.Medium_intervention, actions == Action.Major_intervention, actions == Action.Replace]
        choices = [0, 0, 20, 40, 60, 100]
        self.state[:, OBS.capital] += np.select(conditions, choices, default= self.state[:, OBS.capital])

        rewards = self.calculate_rewards(self.state)

        rewards = -rewards
        assert np.where(self.state[:, OBS.fp] < 0)[0].size == 0, 'Negative fp'

        # Include state in info
        info = {'state': self.state.copy()}  # Ensure a copy of the state is passed

        return self.state, rewards , done, False, info

    # Reset env to initial state
    def reset(self, seed = None):

        if seed is not None:
            self.set_seed(seed)

        self.current_step = 0
        self.state = self.bridges.copy()
        return self.state, {}
    
    def render(self, mode='human'):
        pass

    def _get_obs(self):
        return self.state

    # Calculate rewards (actually costs) for given actions based on given state
    def calculate_rewards(self, state):

        overall_age = np.sum(state[:, OBS.aux_age])

        overall_capital = np.sum(state[:, OBS.capital])

        reward = 0.2*overall_age + 0.8*overall_capital

        return reward

    def _get_obs(self):
        return self.state
    
    def set_seed(self, seed=None):
        # Set seed for reproducibility
        np.random.seed(seed)