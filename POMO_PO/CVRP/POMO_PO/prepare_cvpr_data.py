import argparse
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
from tqdm import tqdm
from docplex.mp.model import Model
import xml.etree.ElementTree as ET
from collections import defaultdict

SCALE = 1e7



def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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
            file.write(" " + str(i + 1) + " " + str(int(node[0] * SCALE)) + " " + str(int(node[1] * SCALE)) + "\n")
        file.write("DEMAND_SECTION\n")

        for i, demand in enumerate(demands):
            file.write(str(i + 1) + " " + str(demand) + "\n")
        file.write("DEPOT_SECTION \n 1 \n -1 \nEOF")
        file.close()


def create_parameter_file(instance_num, working_dir, time_limit=100):

    with open(os.path.join(working_dir, str(instance_num) + ".par"), "w") as file:
        file.write("CPLEX Parameter File Version 12.10.0\n")
        file.write("timelimit = " + str(time_limit) + "\n")
        file.write("mip tolerances mipgap = 1e-4\n")
        file.write("mip tolerances absmipgap = 1e-4\n")
        file.close()


def solve_cvrp_with_cplex(instance_num, nodes, demands, capacity, max_vehicles, working_dir):
    num_nodes = len(nodes)
    distances = squareform(pdist(nodes, metric='euclidean'))
    model = Model(name='CVRP')

    # Decision variables
    x = model.binary_var_matrix(num_nodes, num_nodes, name='x')
    u = model.integer_var_list(num_nodes, lb=0, name='u')

    # Objective function: minimize the total distance
    model.minimize(model.sum(distances[i][j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)))

    # Constraints
    # Each customer is visited exactly once
    for j in range( num_nodes):
        model.add_constraint(model.sum(x[i, j] for i in range(num_nodes) if i != j) == 1)
    # Each customer is left exactly once
    # for i in range(1, num_nodes):
    #     model.add_constraint(model.sum(x[i, j] for j in range(1,num_nodes) if i != j) <= 2)

    # Subtour elimination constraints and capacity constraints
    model.add_constraints(u[i] >= demands[i] for i in range(1, num_nodes))
    model.add_constraints(u[i] <= capacity for i in range(1, num_nodes))
    model.add_constraints(
        u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j] for i in range(1, num_nodes) for j in
        range(1, num_nodes) if i != j)

    # Ensure that no more than max_vehicles leave the depot
    model.add_constraint(model.sum(x[0, j] for j in range(1, num_nodes)) <= max_vehicles)

    # Solve the problem
    solution = model.solve(log_output=True)

    if solution:
        # sol_path = os.path.join(working_dir, str(instance_num) + '.sol')
        solution.export_as_sol(path='./', basename=f'{instance_num}.sol')
        routes = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if x[i, j].solution_value > 0.5:
                    routes.append((i, j))

        tours = []
        visited = set()
        while tours:
            route = []
            current = 0  # Start at the depot
            while True:
                for (i, j) in tours:
                    if i == current:
                        route.append(j)
                        tours.remove((i, j))
                        current = j
                        if j == 0:
                            break
                if current == 0:
                    break
            if route:
                tours.append(route)

        # Write solution to file in VRPLIB format
        vrplib_sol_path = os.path.join(working_dir, str(instance_num) + '_vrplib.sol')
        with open(vrplib_sol_path, 'w') as f:
            for k, route in enumerate(tours):
                f.write(f"Route #{k + 1}: {' '.join(map(str, route))}\n")
            f.write(f"Cost: {solution.objective_value:.4f}\n")

        return tours, solution.objective_value
    else:
        print("No solution found by CPLEX")
        return [], 0


def read_solution_file(instance_num, working_dir, num_nodes):
    file_path = os.path.join(working_dir, str(instance_num) + '.sol')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    tours = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.strip().startswith('<variable name="x_'):
                parts = line.strip().split('"')
                if parts[-2] == '1':
                    _, i, j = parts[1].split('_')
                    tours.append(int(j))
        tours = [0] + tours + [0]  # adding depot at start and end
    return np.array(tours)


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


def load_raw_data(episode=1000000):
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


def parse_cplex_solution(sol_file):
    tree = ET.parse(sol_file)
    root = tree.getroot()

    routes = defaultdict(list)
    for var in root.findall('.//variable'):
        name = var.get('name')
        value = var.get('value')
        if name.startswith('x_') and float(value) == 1.0:
            _, i, j = name.split('_')
            i, j = int(i), int(j)
            routes[i].append(j)

    cost = float(root.find('.//header').get('objectiveValue'))

    return routes, cost


def extract_tours(routes):
    tours = []
    visited = set()
    for start_node in routes[0]:  # 从仓库出发的节点
        if start_node not in visited:
            tour = [start_node]
            current_node = start_node
            visited.add(current_node)
            while current_node in routes and routes[current_node]:
                next_node = routes[current_node].pop()
                if next_node in visited:
                    break
                tour.append(next_node)
                visited.add(next_node)
                current_node = next_node
            tours.append(tour)
    return tours


def write_vrplib_solution(tours, cost, output_file):
    with open(output_file, 'w') as f:
        for route_num, tour in enumerate(tours, start=1):
            f.write(f'Route #{route_num}: ')
            f.write(' '.join(map(str, tour)))
            f.write('\n')
        f.write(f'Cost: {cost}\n')


if __name__ == '__main__':
    num_instances = 1
    num_nodes = 20
    capacity = 1
    max_vehicles = 2
    working_dir = ''
    num_runs = 10
    time_limit = 600
    seed = 777
    instance = 0
    demand_scaler = 1

    if num_nodes == 10:
        demand_scaler = 20
    elif num_nodes == 20:
        demand_scaler = 30
    elif num_nodes == 50:
        demand_scaler = 40

    parser = argparse.ArgumentParser(description='CVRP instance generator and solver')
    parser.add_argument('--num_instances', type=int, default=1, help='Number of instances to generate and solve')
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes per instance')
    parser.add_argument('--capacity', type=int, default=1, help='Vehicle capacity')
    parser.add_argument('--working_dir', type=str, default='', help='Working directory')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs per instance')
    parser.add_argument('--time_limit', type=int, default=100, help='Time limit for CPLEX solver')
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for instance in range(args.num_instances):
        nodes = np.random.rand(args.num_nodes, 2)
        demands = np.random.randint(1, 10, size=args.num_nodes) / demand_scaler
        demands[0] = 0

        create_problem_file(instance, nodes, demands, args.capacity, args.working_dir)
        create_parameter_file(instance, args.working_dir, args.time_limit)

        for run in range(args.num_runs):
            tours, cost = solve_cvrp_with_cplex(instance, nodes, demands, args.capacity, max_vehicles, args.working_dir)
            print("Run", run, ":", tours)

        try:
            solution_tours = read_solution_file(instance, args.working_dir, args.num_nodes)
            print('Solution from file:', solution_tours)
        except FileNotFoundError as e:
            print(e)

    sol_file = os.path.join(working_dir, f'{instance}.sol')
    output_file = os.path.join(working_dir, f'{instance}_vrplib.sol')

    routes, cost = parse_cplex_solution(sol_file)
    tours = extract_tours(routes)
    write_vrplib_solution(tours, cost, output_file)

    print(f'VRPLIB solution written to {output_file}')

