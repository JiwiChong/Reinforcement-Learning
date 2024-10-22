import os
import re
import gym
from gym import spaces
import pickle
import collections
from tqdm import tqdm
import datetime
import itertools
import random
import time
import math
import numpy as np
import pandas as pd
import argparse
# from env.env import InventoriesInKorea as Env
from ast import literal_eval
# from utils import str2bool, make_env, make_env_test

import warnings
warnings.filterwarnings('ignore')


class State(object):
    def __init__(self, warehouse_num, T, demand_history, t = 6): 
        # t = 6 so the demand data starts from this index (remember we are using SMA of last 7 timesteps at each step)
        self.warehouse_num = warehouse_num
        self.factory_stock = 0
        self.warehouse_stock = np.repeat(0, warehouse_num)
        self.demand_history = demand_history
        self.day = 0 
        self.T = T
        self.t = t

    def to_array(self):
        return np.concatenate(([self.factory_stock], self.warehouse_stock, self.demand_history)) # state vector without weekday index

    def stock_levels(self):
        return np.concatenate( ([self.factory_stock], self.warehouse_stock) )

class Action(object):
    def __init__(self, warehouse_num):
        self.production_level = 0
        self.shippings_to_warehouses = np.zeros(warehouse_num)

class InventoriesInKorea(gym.Env):
    def __init__(self, args, mode='train', episode_length=365, train_test_split=0.2, out_csv_name=None, is_mo=False):

        self.T = episode_length  # episode duration
        self.train_test_split = train_test_split
        self.warehouse_num = 6  # number of warehouse
        self.unit_price = 100  # unit price in dollars
        self.unit_cost = 10  # unit cost in dollars, 40
        self.storage_cost_param = 0.05     # Storage cost per unit (previous value: 2.0)
        self.dem_cost_param = 1.0        # Demand cost per unit for not meeting up demand (previous_value = 10.0)
        self.discard_cost_param = 1.0     # Discard cost per unit for exceeding storage capacity (previous value 4.0)
        
        self.demand_dataset = pd.read_csv(f'{args.main_dir}/entire_region_sales.csv')
        
        self.MA_type = 'SMA'
        self.history_length = 7
        self.alpha = 0.4
        self.regions = ['Kangwon', 'Kyonggi','Kyongsang', 'Seoul', 'Jeonla', 'Chungcheong']
        self.storage_capacities = np.array([2000, 300, 300, 300, 300, 300, 300])

        # storage costs at the factory and each warehouse, dollars per unit
        self.storage_costs = np.array([2., 2.])

        # transportation costs for each warehouse, dollars per unit
        self.transporation_costs = np.array([5.])

        self.penalty_unit_cost = 100  

        
        self.mode = mode
        self.info = {}
        self.out_csv_name = out_csv_name
        self.metrics = []
        self.run = 0
        self.step_index = 0
        self.is_mo = is_mo
        if self.is_mo:
            self.reward = np.zeros(4)
        else:
            self.reward = 0

        self.reset()
      
        # action_space & observation_space configuration
        # action space of 0~1 was mainly for convenience and to make it more compact and easier for the agent to learn.
        self.action_space = spaces.Box(low=0, high=1.0, shape=(self.warehouse_num+1, ), dtype=np.float32)
        # observation space can also be changed to -1.0 to 1.0 to make the agent learn better (This part hasn't been addressed yet)
        self.observation_space = spaces.Box(low= -10000, high= 10000, shape=(49,), dtype=np.float32)
    
    # method for  the moving average, might be needed or not depending on how the env is assembled
    def demands_data(self):
        demand_dataset = self.demand_dataset
        demand_dataset = demand_dataset[[n for n in demand_dataset.columns if n.endswith('sales')]]  

        return demand_dataset.round()
    
    def reset(self):
        # demand history list: used be length of 4, but now it would be 1 if we are using the moving average
        demands = self.demands_data()
        self.demands = demands[[n for n in demands.columns if n.endswith('sales')]]

        if self.mode =='train':
            # We are starting index of random starting day from SMA window size so that we can use actual SMA of past in the states
            self.rand_starting_day = np.random.randint(self.history_length, 
                                                       round(len(self.demands) * (1 - self.train_test_split)) - self.T,
                                                       1)[0]
        else:
            self.rand_starting_day = round(len(self.demands) * (1 - self.train_test_split)) if self.T == 365 else np.random.randint(round(len(self.demands) * (1 - self.train_test_split)), # shouldn't this be -1?
                                                       round(len(self.demands) - self.T), 1)[0]
                                                   

        self.t = 0  # current time step value
        
        # We are considering the moving average of past demand values excluding the depaand of current time step

        demand_history = np.hstack([self.demands.iloc[self.rand_starting_day-self.history_length : self.rand_starting_day,i].values.flatten() for i in range(self.history_length-1)])

        self.state = State(self.warehouse_num, self.T, list(demand_history))
        
        # Random initial states for factory and warehouses
        self.state.factory_stock= np.random.randint(0, self.storage_capacities[0], size=1)[0] #self.storage_capacities[0]
        self.state.warehouse_stock=np.random.randint(0, self.storage_capacities[1], size=6) #/self.storage_capacities[0]
        
        if self.mode == 'test':
            np.random.seed(123)
            self.state.factory_stock= np.random.randint(0, self.storage_capacities[0], size=1)[0] #self.storage_capacities[0]
            self.state.warehouse_stock=np.random.randint(0, self.storage_capacities[1], size=6) #/self.storage_capacities[0]
        return self.state.to_array()
    
    def sample_action(self):
        return np.random.rand(self.action_space.shape[0])
    
    def step(self, action):  # action vector: [prod, transw1, transw2, ... , transw6]
        if self.t >= self.T:  # was >= originally
            return self.state.to_array(), self.reward, self.t >= self.T, self.info   # was >= originally
        
        # action check, 6 Warehouse Check.
        action = action * self.storage_capacities    # this one would be the more appropriate one 
        demands = list(self.demands.iloc[self.rand_starting_day])

        # Total original transportation amount, given by the RL agent 
        total_shipment = sum(action[1:])

        # Actual shipment in list of zeros to be filled: 
        real_transported = [0 for i in range(self.warehouse_num)]

        # Scale the shipments down in case total shipments exceed factory stocks. Otherwise emit what the agent performs as actions
        # discard cost can be incurred right above here if factory capacity > self.state.factory_stock+action[0]
        
        if total_shipment > 0:
            if total_shipment > min(self.state.factory_stock+action[0], self.storage_capacities[0]):
                for i in range(self.warehouse_num):
                    # Example: in case the factory stock + production is 2200, we can only use 2000 while 200 must be discarded 
                    real_transported[i] += int(action[i+1] * min(self.state.factory_stock+action[0], self.storage_capacities[0]) / total_shipment)
            else:
                for i in range(self.warehouse_num):
                    real_transported[i] += action[i+1]

        if self.is_mo:  # Ignore this part 
            self.reward_shipment = 1.0
        else:
            self.reward = 1.0

        # Factory discard cost
        factory_discard_cost = max((self.state.factory_stock + action[0]) - self.storage_capacities[0], 0)
        # penalty added for exceeding factory capacity
        self.reward_factory_discard_cost = 0.0
        if int(factory_discard_cost) > 0:
            if self.is_mo:  # Ignore this part 
                self.reward_factory_discard_cost = -1.0 # Ignore this part 
            else:
                self.reward -= 1.0

        warehouse_discard_cost = [0 for i in range(self.warehouse_num)]
        for w in range(self.warehouse_num):  # ****
            # Original form:
            #             warehouse_discard_cost[w] += max(self.state.warehouse_stock[w] + real_transported[w] - np.minimum(demands[w], self.state.warehouse_stock[w])- self.storage_capacities[w + 1], 0)
            warehouse_discard_cost[w] += max(
                self.state.warehouse_stock[w] + real_transported[w] - self.storage_capacities[w + 1], 0)
        
        # penalty added for exceeding warehouse capacities
        self.reward_warehourse_discard_cost = 0.0
        if sum(warehouse_discard_cost) > 0:
            if self.is_mo:  # Ignore this part 
                self.reward_warehourse_discard_cost = -1.0  # Ignore this part 
            else:
                self.reward -= 1.0 
                
        total_discard_cost = (factory_discard_cost + sum(warehouse_discard_cost))  # total amounts that have been discarded! 
        
        '''In terms of discard cost, factory_discard_cost & warehouse_discard_cost have to be saved. If both are >0, add them to cost'''
        
        '''Reward function assemblage:'''
        opening_storage = np.maximum(self.state.stock_levels(), np.zeros(self.warehouse_num + 1))

        opening_storage[0] = min(action[0] + opening_storage[0], self.storage_capacities[0])
        
        for i in range(self.warehouse_num):
            opening_storage[i+1] = min(max((self.state.warehouse_stock[i] + real_transported[i]),0), self.storage_capacities[i+1])
            
        closing_storage = np.copy(opening_storage)
            
        closing_storage[0] -= sum(real_transported)
        for i in range(self.warehouse_num):
            closing_storage[i+1] = max((closing_storage[i+1] - demands[i]),0)

        total_storage_cost = np.mean(np.vstack((opening_storage, closing_storage)), axis=0)
      

        total_storage_cost_ = np.where(self.storage_capacities < total_storage_cost, self.storage_capacities, total_storage_cost)
        
        # storage cost penalties added to the reward function
        if self.is_mo:
            self.reward_storage_cost = -np.around(np.array([np.sum(total_storage_cost_/self.storage_capacities)]))[0]
        else: 
            self.reward -= np.around(np.array([np.sum(total_storage_cost_/self.storage_capacities)]))[0]

        '''In terms of storage cost, total_storage_cost_ has to be saved. '''
       
        # Cost for not meeting demand. Also known as stockout cost, only ocurring at the warehouses!
        demand_cost = [0 for i in range(self.warehouse_num)] 
        for i in range(self.warehouse_num):
            demand_cost[i] += max(demands[i] - (self.state.warehouse_stock[i] + real_transported[i]), 0)
        
        '''In terms of stockout cost, demand_cost has to be saved. '''
        demand_cost_reward = 0
        for d_c in demand_cost:
            if d_c > 0:
                self.reward -= 1.0
                demand_cost_reward -= 1.0
        self.reward_demand_cost = demand_cost_reward

        if self.is_mo:
            self.reward = np.array([self.reward_shipment,self.reward_factory_discard_cost,self.reward_warehourse_discard_cost,self.reward_storage_cost,self.reward_demand_cost])
        '''Reward function overall :'''
        # each reward function component multipled by the monetary factors
        all_t_storages = np.around(total_storage_cost_)  # <class 'numpy.ndarray'>
        all_t_stockouts = np.around(np.array(demand_cost))   # <class 'list'>  
        all_t_factory_overstocks = round(factory_discard_cost)  # <class 'numpy.float64'>
        all_t_retailer_overstocks = np.around(np.array(warehouse_discard_cost))  # <class 'list'>
        all_real_transported = np.around(np.array(real_transported))
        
        '''Next state transition:'''
        self.state.factory_stock = round(min(self.state.factory_stock + action[0] - sum(real_transported), self.storage_capacities[0]))
        # recent change in warehouse stock 
        self.state.warehouse_stock = np.round(np.minimum(np.maximum(self.state.warehouse_stock + real_transported - demands, np.zeros(6)), self.storage_capacities[1:]))
        # previous warehouse stock calculation
        self.t += 1
        self.rand_starting_day += 1
       
        self.state.demand_history = list(np.hstack([self.demands.iloc[self.rand_starting_day-self.history_length : self.rand_starting_day,i].values.flatten() for i in range(self.history_length-1)]))
        self.info['info'] = [demands, all_t_storages, all_t_stockouts, all_t_factory_overstocks, all_t_retailer_overstocks,all_real_transported, round(action[0])]
        done = self.t >= self.T
        info = self._compute_step_info(demands, all_t_storages, all_t_stockouts, all_t_factory_overstocks, all_t_retailer_overstocks, all_real_transported, round(action[0]), self.reward)
        self.metrics.append(info)
#         print('run:', self.run)

        if done:
            self.save_info(self.out_csv_name, self.run)
            self.run += 1
            self.metrics = []
            self.step_index = 0

        # previously reward was being divided by self.T before returning
        return self.state.to_array(), self.reward, done, self.info   # was >= originally

    def _compute_step_info(self, reward, demands, all_t_storages, all_t_stockouts, all_t_factory_overstocks, all_t_retailer_overstocks, all_real_transported, production):
        return {
            'Demands': demands,
            'all_t_storages': all_t_storages,
            'all_t_stockouts': all_t_stockouts,
            'all_t_factory_overstocks': all_t_factory_overstocks,
            'all_t_retailer_overstocks': all_t_retailer_overstocks,
            'all_real_transported': all_real_transported,
            'production': production,
            'Rewards': np.sum(reward) 

        }

    def save_info(self, out_csv_name, run):
        if out_csv_name is not None:
#             pickle.dump(self.metrics, open(out_csv_name + '_run{}'.format(run) + '.pickle', "wb"))
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)
            

    def close(self):
        pass

def make_env(args, rank):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = InventoriesInKorea(mode='train', episode_length=365, train_test_split=0.2, out_csv_name=f'{args.main_dir}/results/{args.method}_results/Total_penalty/run_{args.run_num}/info_df_{rank}')
        return env
    
    return _init


def make_env_test(args, rank):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        # make sure you change model's name and training num here! 
        env = InventoriesInKorea(mode='test', episode_length=365, train_test_split=0.2, out_csv_name=f'{args.main_dir}/results/{args.method}_results/Total_penalty/run_{args.run_num}/info_df_{rank}')
        return env
    
    return _init