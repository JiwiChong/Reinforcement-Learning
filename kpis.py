import numpy as np

def total_costs(total_test_stocks_plus_received, total_test_demands):
    all_costs = []
    for i in range(total_test_stocks_plus_received.shape[0]):
        storage_cost = max(total_test_stocks_plus_received[i] - total_test_demands[i], 0) * 0.05
        stockout_cost = max(total_test_demands[i] - total_test_stocks_plus_received[i], 0) * 1.0
        # print(storage_cost, stockout_cost)
        total_cost = storage_cost + stockout_cost
        all_costs.append(total_cost)
    
    return all_costs

def sell_through_rate(stocks_plus_transports, demands, data_portion=None, epi_len=None):
    # New one:
    nominator = np.array([(np.sum(np.minimum(stocks_plus_transports[i], demands[i]))) for i in range(epi_len)])
    nominator_ = np.array([val for val in list(nominator) if not np.isinf(val)])
    nominator__ = np.array([val for val in list(nominator_) if not np.isinf(val)])[:round(data_portion*epi_len)]

    denominator = np.array([(np.sum(stocks_plus_transports[i])) for i in range(365)])
    denominator_ = np.array([val for val in list(denominator) if not np.isinf(val)])
    denominator__ = np.array([val for val in list(denominator_) if not np.isinf(val)])[:round(data_portion*epi_len)]
    sell_through_rate = (np.sum(nominator__)/np.sum(denominator__)) * 100
    # print(type(sell_through_rate))
    return sell_through_rate

def service_level(stocks_plus_transports, demands, data_portion=None, epi_len=None):
    ep_ratios = []
    demands_ = []
    vals = []
    ratios = []
#     for d, s in zip(demands[1:,:], stocks_plus_transports[1:,:]):  # originally
    for d, s in zip(demands, stocks_plus_transports):
#         print(d,s)
        val = np.minimum(d, s)
        vals.append(val)
        demands_.append(d)
    service_level = (np.sum(np.array(vals)[:round(epi_len*data_portion)]) / np.sum(np.array(demands)[:round(epi_len*data_portion)])) * 100
    return service_level

def inventory_to_sales_ratio(stocks_plus_received, demands, data_portion, unit_price=100):
    """
    The mathematical formula for inventory to sales ratio is referenced from the following link:
    
    https://studyfinance.com/inventory-to-sales-ratio/
    
    Inventory to sales ratio = Average Inventory / Net Sales
    
    whereas 
    
    Average Inventory = (Beginning Inventory + Ending Inventory) / 2
    """
    
    # Calculates average inventory for each warehouse
    beginning_inventory = stocks_plus_received[0,:] * unit_price
    
    ending_stocks = stocks_plus_received[-1,:]
    ending_sales = demands[-1,:]
    
    ending_inventory = []
    for stock, sales in zip(ending_stocks, ending_sales):
        if stock > sales:
            ending_inventory.append(stock - sales)
        else:
            ending_inventory.append(0)
    
    ending_inventory = np.array(ending_inventory) * unit_price
    
    
    avg_inventory = (np.sum(beginning_inventory) + np.sum(ending_inventory)) / 2
    
    # Calculating net sales volume for each warehouse
    net_sales = np.zeros(demands.shape[1])
    
    for i in range(demands.shape[0]):
        for j in range(net_sales.shape[0]):
            net_sales[j] += min(demands[i][j], stocks_plus_received[i][j])
    
    net_sales = net_sales * unit_price
    
    inventory_to_sales = avg_inventory / np.sum(net_sales)
    
    return inventory_to_sales