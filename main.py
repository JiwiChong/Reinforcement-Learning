import argparse
import numpy as np
from torch import nn as nn
from stable_baselines3 import PPO, A2C, TD3, SAC
from sb3_contrib import TRPO
from inventory_env import InventoriesInKorea, make_env
from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    parser.add_argument("-method", dest="method", type=str, required=False, help="Which RL model to run")
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--num_cpu', type=str, default=8, help='Dataset to be used')
    parser.add_argument('--run_num', type=str, help='run number')

    args = parser.parse_args()

    if args.method == 'random':
        env = InventoriesInKorea(mode='train', episode_length=365, train_test_split=0.2, out_csv_name=f'{args.main_dir}results/SB3_results/random/Total_penalty/run_{args.run_num}/info_df/rewards')
        env.reset()
        for i in range(370000):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                env.reset()
    else:
        if args.method == 'ppo':
            # Create the vectorized environment
            env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            model = PPO('MlpPolicy', env, gamma=0.6, n_steps=8, batch_size=64, learning_rate=0.0013711273108357805, clip_range=0.2, n_epochs=10,
                        ent_coef=3.025916547920685e-07, gae_lambda=0.9, vf_coef=0.2890829508607011,  max_grad_norm=5,
                        policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])]),  verbose=0)
            model.learn(total_timesteps=3000000)
            env.save(f"{args.main_dir}/results/PPO_results/PPO/Total_penalty/run_{args.run_num}/saved_env/vec_normalize_inv.pkl")
            model.save(f"{args.main_dir}/results/PPO_results/PPO/run_{args.run_num}/saved_model/PPO_Model")
            
        elif args.method == 'a2c':
            # Create the vectorized environment
            env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            model = A2C("MlpPolicy", env, gamma=0.4, normalize_advantage=True, n_steps=64, learning_rate=0.00016165809152017579,
                        ent_coef=6.52767763575326e-05, gae_lambda=0.99, vf_coef=6.52767763575326e-05, max_grad_norm=0.9, 
                        policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])]),  verbose=0)
            model.learn(total_timesteps=365000)
            env.save(f"{args.main_dir}/results/A2C_results/A2C/run_{args.run_num}/saved_env/vec_normalize_inv.pkl")
            model.save(f"{args.main_dir}/results/A2C_results/A2C/run_{args.run_num}/saved_model/A2C_Model")

        elif args.method == 'td3':
            # Create the vectorized environment
            env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3("MlpPolicy", env, train_freq=2000, verbose=0)
            model.learn(total_timesteps=73000) 
            env.save(f"{args.main_dir}/results/SB3_results/TD3/Total_penalty/run_{args.run_num}/saved_env/vec_normalize_inv.pkl")
            model.save(f"{args.main_dir}/results/SB3_results/TD3/Total_penalty/run_{args.run_num}/saved_model/TD3_Model")
        
        elif args.method == 'sac':
            # Create the vectorized environment
            env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            model = SAC('MlpPolicy', env, gamma=0.5, learning_rate=0.0003, batch_size=32, learning_starts=0, train_freq=8, buffer_size=100000,
                        tau=0.005, verbose=0)
            model.learn(total_timesteps=3000000) # 3000000
            env.save(f"{args.main_dir}/results/SB3_results/TD3/run_{args.run_num}/saved_env/vec_normalize_inv.pkl")
            model.save(f"{args.main_dir}/results/SB3_results/TD3/run_{args.run_num}/saved_model/SAC_Model")