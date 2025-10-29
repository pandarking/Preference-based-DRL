"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
SCALE = 1e7
import torch
from tqdm import tqdm


def create_problem_file(instance_num, nodes, demands, capacity, working_dir):
    with open(os.path.join(working_dir, str(instance_num) + ".vrp"), "w") as file:
        file.write("NAME : " + str(instance_num) + "\n")
        file.write("COMMENT : generated instance No. " + str(instance_num) + "\n")
        file.write("TYPE : CVRP\n")
        file.write("DIMENSION : " + str(len(nodes)) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D \n")
        file.write("CAPACITY : " + str(capacity) + " \n")
        file.write("NODE_COORD_SECTION\n")

        for i, node in enumerate(nodes):
            file.write(" " + str(i+1) + " " + str(int(node[0] * SCALE)) + " " + str(int(node[1] * SCALE)) + "\n")
        file.write("DEMAND_SECTION\n")
        for i, demand in enumerate(demands):
            file.write(str(i+1) + " " + str(demand) + "\n")
        file.write("DEPOT_SECTION \n 1 \n -1 \nEOF ")
        file.close()


def create_parameter_file(instance_num, working_dir, num_runs, time_limit):
    with open(os.path.join(working_dir, str(instance_num) + ".par"), "w") as file:
        file.write("PROBLEM_FILE = " + os.path.join(working_dir, str(instance_num) + ".vrp\n"))
        file.write("RUNS = " + str(num_runs) + "\n")
        if time_limit > 0:
            file.write("TIME_LIMIT = " + str(time_limit) + "\n")
        file.write("TOUR_FILE = " + os.path.join(working_dir, str(instance_num) + ".sol\n"))


def read_solution_file(instance_num, working_dir, num_nodes):
    with open(os.path.join(working_dir, str(instance_num) + ".sol"), "r") as file:
        lines = file.readlines()
        tours = list()
        for node in lines[6:-2]:
            tours.append(int(node))
        tours.append(1)
        tours = np.array(tours)
        tours = tours - 1
        tours[tours > num_nodes] = 0
    return tours


def reorder(coordinates, demands, capacity, all_tours):
    tours, subtour = list(), list()

    for node_idx in all_tours[1:]:
        if node_idx == 0:
            tours.append(subtour)
            subtour = list()
        else:
            subtour.append(node_idx)

    reformated_tour, remaining_capacities = list(), list()
    distances, capacities = list(), list()
    for tour in tours:
        tour_capacity = capacity
        for node_idx in tour:
            tour_capacity -= demands[node_idx]
        capacities.append(tour_capacity)

    tour_idxs = np.argsort(capacities)

    for num_tour in tour_idxs:
        reformated_tour.extend(tours[num_tour])
        first = True
        for node in tours[num_tour]:
            if first:
                remaining_capacities.append(capacity - demands[node])
                first = False
            else:
                remaining_capacities.append(remaining_capacities[-1] - demands[node])

    # add depot at the beginning and the end
    remaining_capacities = [capacity] + remaining_capacities + [capacity]
    tour = [0] + reformated_tour + [0]
    tour = np.array(tour)
    via_depot = np.array([0.] * len(tour))
    via_depot[0] = 1.

    for i in range(1, len(remaining_capacities) - 1):
        if remaining_capacities[i] > remaining_capacities[i - 1]:
            via_depot[i] = 1.

    if reorder:
        coordinates = coordinates[tour]
        demands = demands[tour]

    return coordinates, demands, remaining_capacities, via_depot

def load_raw_data( episode=1000000):
    def tow_col_nodeflag(node_flag):
        tow_col_node_flag = []
        V = int(len(node_flag) / 2)
        for i in range(V):
            tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
        return tow_col_node_flag

    # Because the dataset is too large, I split it into two reads

    raw_data_nodes = []
    raw_data_capacity = []
    raw_data_demand = []
    raw_data_cost = []
    raw_data_node_flag = []
    for line in tqdm(open('vrp200_test_lkh.txt', "r").readlines()[0:episode], ascii=True):
        line = line.split(",")

        depot_index = int(line.index('depot'))
        customer_index = int(line.index('customer'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demand'))
        cost_index = int(line.index('cost'))
        node_flag_index = int(line.index('node_flag'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, capacity_index, 2)]

        loc = depot + customer
        capacity = int(float(line[capacity_index + 1]))
        if int(line[demand_index + 1]) == 0:
            demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        else:
            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

        cost = float(line[cost_index + 1])
        node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

        node_flag = tow_col_nodeflag(node_flag)

        raw_data_nodes.append(loc)
        raw_data_capacity.append(capacity)
        raw_data_demand.append(demand)
        raw_data_cost.append(cost)
        raw_data_node_flag.append(node_flag)

    raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
    # shape (B,V+1,2)  customer num + depot
    raw_data_capacity = torch.tensor(raw_data_capacity, requires_grad=False)
    # shape (B )
    raw_data_demand = torch.tensor(raw_data_demand, requires_grad=False)
    # shape (B,V+1) customer num + depot
    raw_data_cost = torch.tensor(raw_data_cost, requires_grad=False)
    # shape (B )
    raw_data_node_flag = torch.tensor(raw_data_node_flag, requires_grad=False)
    # shape (B,V,2)

    return raw_data_nodes, raw_data_demand


if __name__ == '__main__':
    num_instances = 1
    num_nodes = 20
    capacity = 30
    working_dir = ''
    num_runs = 10
    time_limit = 100
    lkh_exec = 'LKH-3.exe'
    seed = 777
    output_filename = 'file.npz'
    reorder = False

    np.random.seed(seed)
    all_coords, all_demands, all_capacities, all_remaining_capacities = list(), list(), list(), list()
    all_via_depots, all_tour_lens = list(), list()
    for instance_num in range(num_instances):
        coords = np.random.rand(num_nodes + 1, 2)
        demands = np.array([0] + np.random.randint(1, 10, num_nodes).tolist())
        create_problem_file(instance_num, coords, demands, capacity, working_dir)
        create_parameter_file(instance_num, working_dir, num_runs=num_runs,   time_limit=time_limit)
        lkh_cmd = os.path.join(lkh_exec) + ' ' + os.path.join(working_dir, str(instance_num) + ".par")
        print('lkh_cmd:', lkh_cmd)
        os.system(lkh_cmd)   #执行LKH.exe，形成sol路径文件
        tours = read_solution_file(instance_num, working_dir, num_nodes)

        # add first node to the end
        coords = coords.tolist()
        coords.append(coords[0])
        demands = demands.tolist()
        demands.append(demands[0])
        coords = np.array(coords)
        demands = np.array(demands)

        adj_matrix = squareform(pdist(coords, metric='euclidean'))
        tour_len = sum([adj_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)])

        if reorder:
            coords, demands, remaining_capacities, via_depots = reorder(coords, demands, capacity, tours)
        else:
            remaining_capacities, via_depots = None, None

        all_coords.append(coords)
        all_demands.append(demands)
        all_remaining_capacities.append(remaining_capacities)
        all_via_depots.append(via_depots)
        all_tour_lens.append(tour_len)
        all_capacities.append(capacity)

    capacities = np.stack(all_capacities)
    coords = np.stack(all_coords)
    demands = np.stack(all_demands)
    if reorder:
        remaining_capacities = np.stack(all_remaining_capacities)
        via_depots = np.stack(all_via_depots)
        np.savez_compressed(output_filename, capacities=capacities, coords=coords, demands=demands,
                            remaining_capacities=remaining_capacities, via_depots=via_depots, reorder=True)
    else:
        tour_lens = np.stack(all_tour_lens)
        np.savez_compressed(output_filename, capacities=capacities, coords=coords, demands=demands,
                            tour_lens=tour_lens, reorder=False)

    print("Result saved in " + output_filename)
