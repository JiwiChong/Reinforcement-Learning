from inventory_env import InventoriesInKorea, make_env_test
import numpy as np
import argparse
from stable_baselines3 import A2C, TD3, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from kpis import total_costs, sell_through_rate, service_level, inventory_to_sales_ratio

def test_inventories(args):
    path = f"{args.main_dir}/results/{args.method}_results/Total_penalty/run_{args.run_num}/saved_env/vec_normalize_inv.pkl"  # path of saved env
    m_path = f"{args.main_dir}/results/{args.method}_results/Total_penalty/run_{args.run_num}/saved_model/{args.method}_Model"  # path of saved model

    #env creation
    env_test = SubprocVecEnv([make_env_test(i) for i in range(args.num_cpu)])
    env_test = VecNormalize.load(path+"vec_normalize_inv.pkl", env_test)

    #  do not update them at test time
    env_test.training = False
    # reward normalization is not needed at test time
    env_test.norm_reward = False

    if args.method == 'sac':
         model = SAC.load(m_path+'SAC_Model', env=env_test)
    elif args.method == 'a2c':
         model = A2C.load(m_path+'A2C_Model', env=env_test)
    elif args.method == 'trpo':
         model = TRPO.load(m_path+'TRPO_Model', env=env_test)
    elif args.method == 'td3':
         model = TD3.load(m_path+'TD3_Model', env=env_test)
    
    state = env_test.reset()
    done=False
    N = 365
    num_warehouses = 6
    state = env_test.reset()
    done=[False for _ in range(args.num_cpu)] 
    stocks_plus_received=[[] for _ in range(args.num_cpu)]
    demands=[[] for _ in range(args.num_cpu)]

    while not done[0]:
        action, _states = model.predict(state)
        next_state, r, done, info = env_test.step(action)
        for i in range(args.num_cpu):
            stocks_plus_received[i].append(np.array(state[i][1:7]) + np.array(info[i]['info'][-2]))
            demands[i].append(info[i]['info'][0]) # keeps appending new demand to the demands array on the top 

        if done.all():
            state = None
            break
        else:
            state = next_state

    stocks_plus_received = np.array(stocks_plus_received).mean(axis=0)
    demands = np.array(demands).mean(axis=0)

    return stocks_plus_received, demands

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    parser.add_argument("-method", dest="method", type=str, required=False, help="Which RL model to run")
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--num_cpu', type=str, default=8, help='Dataset to be used')
    parser.add_argument('--run_num', type=str, help='run number')

    args = parser.parse_args()

    stocks_plus_received, demands = test_inventories(args)
    total_test_stocks_plus_received = np.sum(stocks_plus_received, axis=1)
    total_test_demands = np.sum(demands, axis=1)

    all_costs = total_costs(total_test_stocks_plus_received, total_test_demands)

    data_portions = [0.16, 0.25, 0.5, 0.75, 1.0]
    epi_len = 365
    for data_portion in data_portions:
        print(f'Data portion is {data_portion}')
        print(f'Total cost is {round(sum(all_costs[:round(epi_len*data_portion)]),2)}')
        print(f'Sell through rate is {round(sell_through_rate(stocks_plus_received, demands, data_portion=data_portion, epi_len=365),2)}')
        print(f'Invenory to sales ratio = {round(inventory_to_sales_ratio(stocks_plus_received[:round(epi_len*data_portion)], demands[:round(epi_len*data_portion)], data_portion=data_portion, unit_price=100), 5)}')
        print(f'Service level is {round(service_level(stocks_plus_received, demands, data_portion=data_portion, epi_len=365),2)}')
        print('------------------------------------------')


# Commands:
# python main.py --method sac --main_dir ./Reinforcement-Learning --num_cpu 8 --run_num (# of run)