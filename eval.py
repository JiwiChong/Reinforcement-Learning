from inventory_env import InventoriesInKorea
import numpy as np
import argparse
from kpis import total_costs, sell_through_rate, service_level, inventory_to_sales_ratio

def test_inventories(args):
    env = InventoriesInKorea(mode='test', episode_length=365, train_test_split=0.2, out_csv_name=f'{args.main_dir}/results/SB3_results/random/Total_penalty/run_{args.run_num}/test_info_df/rewards')

    #test the Random agent
    state = env.reset()
    done=False
    N = 365
    num_warehouses = 6
    stocks_plus_received=[]
    demands=[]
    amount_produced = []

    while not done:
        action = env.action_space.sample()
        next_state, r, done, info = env.step(action)
        stocks_plus_received.append(np.array(state[1:7]) + np.array(info['info'][-2]))
        demands.append(info['info'][0]) # keeps appending new demand to the demands array on the top 
        amount_produced.append(info['info'][-1])
        if done:
            state = None
            break
        else:
            state = next_state  

    stocks_plus_received = np.array(stocks_plus_received)
    demands = np.array(demands)

    return stocks_plus_received, demands

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