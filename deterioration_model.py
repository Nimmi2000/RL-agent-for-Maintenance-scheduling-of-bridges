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

        # Read the csv and create extra metrics for tracking the health condition of the bridges
        df = pd.read_csv(df_path)
        self.bridge_ids = df['bridge_id'].to_numpy()
        
        bridge_features = df[['traffic_intensity', 'latitude','longitude']].copy()
        bridge_features['age'] = (datetime.datetime.now().year - df['last_replacement']).abs()
        bridge_features['aux_age'] = bridge_features['age'].copy()
        bridge_features['aux_age'] = bridge_features['aux_age'].replace(0, 1) 
        bridge_features['failure_prob'] = 0.0
        bridge_features['reliability'] = 0.0
        bridge_features['capital_invested'] = 0.0

        self.bridges = bridge_features.to_numpy().astype('float64')

        self.num_bridges = len(self.bridges)
        self.observation_shape = self.bridges.shape
        self.n_actions = 3

        self.bridges[:, OBS.sf] = self.calculate_sf(self.bridges, Mu=80, beta = 2)
        self.bridges[:, OBS.fp] = 1 - self.bridges[:, OBS.sf]

        self.action_space = spaces.MultiDiscrete([6] * 15)
        self.observation_space = spaces.Box(0, np.inf , shape=(len(self.bridges), self.bridges.shape[1]), dtype=np.float64)

    #Weibull model for calculation the survival function
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

        # Impact on aux age based on the actions taken
        conditions = [actions == Action.Nothing, actions == Action.Monitoring, actions == Action.Minor_intervention, actions == Action.Medium_intervention, actions == Action.Major_intervention]
        choices = [1, 1, -20, -40, -60]

        delta_age = np.select(conditions, choices, default=1)
        self.state[:, OBS.aux_age] = np.maximum(1, self.state[:, OBS.aux_age] + delta_age)

        self.state[np.where(actions == Action.Replace)[0], OBS.aux_age] = 1  # Set aux age to 1 for replace actions

        # Calculate cost for each step
        action_costs = np.zeros(self.num_bridges, dtype=np.float32)
        cost_map = {
            Action.Nothing: 0.0,
            Action.Monitoring: 1.0,  # Assign a small cost maybe? Or keep 0.
            Action.Minor_intervention: 20.0,
            Action.Medium_intervention: 40.0,
            Action.Major_intervention: 60.0,
            Action.Replace: 100.0
        }

        # Ensure actions are INT format
        int_actions = np.array(actions).astype(int)
        for i, action_val in enumerate(int_actions):
            action_costs[i] = cost_map.get(Action(action_val), 0.0) # Use Action enum

        # This will not be used for reward, just tracking purpose
        self.state[:, OBS.capital] += action_costs

        # Update survival function
        self.state[:, OBS.sf] = self.calculate_sf(self.state, Mu=80, beta = 2)

        #Calculating Failure probabilities
        self.state[:, OBS.fp] = 1 - self.state[:, OBS.sf]

        rewards = self.calculate_rewards(self.state, action_costs)

        assert np.where(self.state[:, OBS.fp] < 0)[0].size == 0, 'Negative fp'

        # Include state in info
        info = {'state': self.state.copy()} 

        return self.state, rewards , done, False, info

    # Reset env to initial state
    def reset(self, seed = None):

        super().reset(seed = seed)

        self.current_step = 0
        self.state = self.bridges.copy()
        self.state[:, OBS.capital] = 0.0
        self.state[:, OBS.sf] = self.calculate_sf(self.state, Mu=80, beta = 2)
        self.state[:, OBS.fp] = 1 - self.state[:, OBS.sf]
        return self.state, {}
    
    def render(self, mode='human'):
        pass

    def _get_obs(self):
        return self.state

    # Calculate rewards (actually costs) for given actions based on given state
    def calculate_rewards(self, state, action_costs):

        rewards = - (0.5 * np.sum(state[:, OBS.aux_age]) +
                 0.4 * (np.sum(state[:, OBS.fp]) * 100) +
                 0.1 * np.sum(action_costs))

        return rewards
    
    def set_seed(self, seed=None):
        # Set seed for reproducibility
        np.random.seed(seed)