import torch
import json
import pickle
import numpy as np
import torch.nn.functional as F
# import Levenshtein


def get_node_pairs(path):
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]


def get_node_pairs_batch(paths):

    node_start = paths[:, :, :-1]
    node_end = paths[:, :, 1:]

    node_pairs = torch.stack((node_start, node_end), dim=-1)

    return node_pairs

def count_matching_pairs_batch(selected_paths, historys):

    selected_pairs = get_node_pairs_batch(selected_paths)  # ((batch_size, pomo_size, n, 2)
    history_pairs = get_node_pairs_batch(historys)  # ((batch_size, pomo_size, m, 2)

    min_route_length = min(selected_pairs.shape[2], history_pairs.shape[2])
    selected_pairs_trimmed = selected_pairs[:, :, :min_route_length, :]  # (batch_size, pomo_size, min_route_length, 2)
    history_pairs_trimmed = history_pairs[:, :, :min_route_length, :]  # (batch_size, history_size, min_route_length, 2)

    # dimensional extension
    selected_pairs_expanded = selected_pairs_trimmed.unsqueeze(2)  # (batch_size, pomo_size, 1, min_route_length, 2)
    history_pairs_expanded = history_pairs_trimmed.unsqueeze(1)  # (batch_size, 1, history_size, min_route_length, 2)

    # compare node pairs (batch_size, pomo_size, history_size, min_route_length, 2)
    matches = (selected_pairs_expanded == history_pairs_expanded).all(dim=-1)

    matching_pairs_count = matches.any(dim=-1).sum(dim=-1)

    return matching_pairs_count


def count_ad(selected_paths, history_paths):
    """
    Compute the arc difference between each selected_route and the historical path.
    Return: The mean and proportion of the arc difference.
    """
    selected_pairs = get_node_pairs_batch(selected_paths)  # (batch_size, pomo_size, route_length-1, 2)
    history_pairs = get_node_pairs_batch(history_paths)  # (batch_size, history_size, route_length-1, 2)

    max_len = max(selected_pairs.shape[2], history_pairs.shape[2])

    # Pad to the same length
    selected_pairs_padded = F.pad(selected_pairs, (0, 0, 0, max_len - selected_pairs.shape[2]))
    history_pairs_padded = F.pad(history_pairs, (0, 0, 0, max_len - history_pairs.shape[2]))

    # Dimensional extension
    selected_pairs_expanded = selected_pairs_padded.unsqueeze(2)  # (batch_size, pomo_size, 1, max_len, 2)
    history_pairs_expanded = history_pairs_padded.unsqueeze(1)  # (batch_size, 1, history_size, max_len, 2)

    # Calculate the difference matrix
    diff_matrix = (selected_pairs_expanded != history_pairs_expanded).all(dim=-1)  # (batch_size, pomo_size, history_size, max_len)
    # print('diff matrix shape:', diff_matrix.shape)
    # 对 diff_matrix 使用 logical NOT 运算符，找到存在于 selected_pairs 中但不在 history_pairs 中的边对
    # unmatched_matrix = ~diff_matrix  # (batch_size, pomo_size, history_size, max_len)

    # Count the number of mismatched edges
    unmatched_count = diff_matrix.sum(dim=-1)  # (batch_size, pomo_size, history_size)
    # print('unmatched count shape:', unmatched_count.shape)

    avg_diff_count = unmatched_count.float().min(dim=-1)[0]  # (batch_size)
    route_edge_count = selected_pairs.shape[2]
    avg_diff_ratio = avg_diff_count / route_edge_count

    # print('diff count shape:', avg_diff_count.shape)
    # print('diff ratio shape:', avg_diff_ratio.shape)

    return avg_diff_count, avg_diff_ratio


def count_rd(selected_route, history_route):
    """
    Calculate the minimum Route Difference and RD percentage for the predicted path.
    """
    batch_size, pomo_size, route_length = selected_route.shape
    _, history_size, _ = history_route.shape

    min_rd_list = []
    rd_percent_list = []

    for b in range(batch_size):

        selected = [set(route.tolist()) for route in selected_route[b]]
        history = [set(route.tolist()) for route in history_route[b]]

        batch_min_rd = []
        total_stops = sum(len(route) for route in history)

        for sel_route in selected:
            rd_values = [len(sel_route.symmetric_difference(hist_route)) for hist_route in history]
            batch_min_rd.append(min(rd_values))

        avg_min_rd = sum(batch_min_rd) / len(batch_min_rd)
        rd_percent = avg_min_rd / total_stops if total_stops > 0 else 0

        min_rd_list.append(avg_min_rd)
        rd_percent_list.append(rd_percent)

    return torch.tensor(min_rd_list, dtype=torch.float32), torch.tensor(rd_percent_list, dtype=torch.float32)


def trim_zeros(path):
    """ Remove 0 at the beginning and end of the path """
    start = 0
    while start < len(path) and path[start] == 0:
        start += 1

    end = len(path) - 1
    while end >= start and path[end] == 0:
        end -= 1

    return path[start:end + 1] if start <= end else []


def sample_subinstances(batch_size, problem_size, subset_size):
    """
    Randomly draw multiple subinstances from an original_size instance
    """
    depot_xy = torch.rand(size=(1, 1, 2)).cuda()
    node_xy = torch.rand(size=(1, problem_size, 2)).cuda()
    # node_demand = torch.randint(1, 10, size=(1, problem_size))

    if subset_size == 5:
        demand_scaler = 10
    elif subset_size == 10:
        demand_scaler = 20
    elif subset_size == 15:
        demand_scaler = 25
    elif subset_size == 20:
        demand_scaler = 30
    elif subset_size == 29:
        demand_scaler = 35
    elif subset_size == 50:
        demand_scaler = 40
    elif subset_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(1, problem_size), dtype=torch.float32) / float(demand_scaler)
    # print('node_demand:', node_demand)
    # depot_xy = torch.tensor([[30, 40]], dtype=torch.float32)
    # depot_xy = torch.tensor([[0.4000, 0.4000]])
    depot_xy = torch.tensor([[0.0000, 0.0000]])
    # node_xy = torch.tensor([[[37, 52], [49, 49], [52, 64], [31, 62], [52, 33], [42, 41], [52, 41], [57, 58], [62, 42], [42, 57], [27, 68], [43, 67], [58, 48], [58, 27], [37, 69], [38, 46], [61, 33], [62, 63], [63, 69], [45, 35]]], dtype=torch.float32)
    # node_xy = torch.tensor([[[0.2200, 0.2200], [0.3600, 0.2600], [0.2100, 0.4500], [0.4500, 0.3500], [0.5500, 0.2000], [0.3300, 0.3400], [0.5000, 0.5000], [0.5500, 0.4500], [0.2600, 0.5900], [0.4000, 0.6600], [0.5500, 0.6500], [0.3500, 0.5100], [0.6200, 0.3500], [0.6200, 0.5700], [0.6200, 0.2400], [0.2100, 0.3600], [0.3300, 0.4400], [0.0900, 0.5600], [0.6200, 0.4800], [0.6600, 0.1400], [0.4400, 0.1300], [0.2600, 0.1300], [0.1100, 0.2800], [0.0700, 0.4300], [0.1700, 0.6400], [0.4100, 0.4600], [0.5500, 0.3400], [0.3500, 0.1600], [0.5200, 0.2600], [0.4300, 0.2600], [0.3100, 0.7600], [0.2200, 0.5300], [0.2600, 0.2900], [0.5000, 0.4000], [0.5500, 0.5000], [0.5400, 0.1000], [0.6000, 0.1500], [0.4700, 0.6600], [0.3000, 0.6000], [0.3000, 0.5000], [0.1200, 0.1700], [0.1500, 0.1400], [0.1600, 0.1900], [0.2100, 0.4800], [0.5000, 0.3000], [0.5100, 0.4200], [0.5000, 0.1500], [0.4800, 0.2100], [0.1200, 0.3800], [0.1500, 0.5600], [0.2900, 0.3900], [0.5400, 0.3800], [0.5500, 0.5700], [0.6700, 0.4100], [0.1000, 0.7000], [0.0600, 0.2500], [0.6500, 0.2700], [0.4000, 0.6000], [0.7000, 0.6400], [0.6400, 0.0400], [0.3600, 0.0600], [0.3000, 0.2000], [0.2000, 0.3000], [0.1500, 0.0500], [0.5000, 0.7000], [0.5700, 0.7200], [0.4500, 0.4200], [0.3800, 0.3300], [0.5000, 0.0400], [0.6600, 0.0800], [0.5900, 0.0500], [0.3500, 0.6000], [0.2700, 0.2400]]])
#     node_xy = torch.tensor([[
#    [0.35, -0.56], [0.72, -0.58], [0.7, -0.66], [0.45, -0.4], [0.39, -0.4], [0.6, -0.5], [0.42, -0.59], [0.31, -0.46], [0.44, -0.58],
#   [0.45, -0.67], [0.69, -0.46], [0.24, 0], [0.12, -0.04], [0.01, -0.21], [0.03, 0.29], [0.19, -0.13], [0.13, -0.14], [0.25, 0.11], [0.24, 0.23],
#   [0.03, 0.07], [0.23, 0.19], [0.02, -0.07], [0.05, 0.23], [0.32, 0.05], [0.14, 0.25], [-0.16, -0.04], [0.24, 0.17], [0, -0.07], [-0.74, -0.22],
#   [-0.64, -0.24], [-0.71, -0.19], [-0.91, -0.15], [-0.65, -0.14], [-0.91, -0.26], [-0.76, -0.07], [-0.66, -0.04], [-0.87, -0.1], [-0.73, -0.08], [-0.81, -0.01],
#   [-0.82, -0.24], [-0.87, -0.25], [-0.76, -0.25], [-0.75, -0.06], [-0.7, -0.03], [-0.64, -0.22], [-0.66, -0.05], [-0.72, -0.1], [-0.89, -0.03], [-0.86, -0.03],
#   [-0.57, -0.09], [-0.22, -0.36], [-0.44, 0.19], [-0.21, 0.06], [-0.49, -0.04], [-0.68, -0.07], [-0.42, 0.11], [-0.69, 0.03], [-0.49, 0.09], [-0.68, -0.19],
#   [-0.57, -0.07], [-0.61, -0.34], [-0.36, 0.16], [-0.56, 0.02], [-0.67, 0], [-0.17, -0.14], [-0.17, -0.2], [-0.28, -0.26], [-0.7, -0.21], [-0.46, -0.14],
#   [-0.52, 0.36], [-0.33, 0.62], [-0.53, 0.49], [-0.39, 0.59]
# ]])
    node_xy = torch.tensor([[
  [-0.02, -0.05], [0.03, -0.01], [-0.02, 0.07], [0.03, 0], [0.07, -0.03], [-0.05, -0.01], [-0.03, 0], [0, -0.03], [0.04, 0.01],
  [0.04, -0.05], [-0.02, -0.04], [0.05, -0.05], [-0.04, 0.01], [-0.02, 0.03], [0.07, 0.02], [-1.3, 0.8], [-0.6, 0.88], [-1.6, 0.83],
  [-1.4, 0.77], [-2.6, 0.88], [-2.4, 0.79], [-2.5, 0.78], [-1.1, 0.8], [-0.5, 0.88], [-2.1, 0.9], [-0.5, 0.82], [-2, 0.8], [-0.6, 0.91],
  [-2.5, 0.89], [-2.5, 0.74], [-0.8, 0.88], [-2.1, 0.74], [-2.3, 0.74], [-0.5, 0.87], [-0.8, 0.91], [-0.03, -0.08], [-0.3, 0.15], [-0.14, 0.24],
  [-0.12, 0.11], [-0.01, 0.07], [0.02, 0.14], [-0.26, 0.15], [-0.36, 0.02], [-0.29, 0.07], [-0.01, 0.28], [-0.07, 0.17], [-0.1, 0.31], [-0.2, -0.02],
  [-0.33, 0.2], [-0.26, 0], [-0.13, 0.09], [-0.08, 0.12], [-0.35, -0.08], [-0.33, 0.23], [0.1, -0.7], [0.08, -0.95], [-0.12, -1.01], [0.06, -1.13],
  [-0.17, -0.76], [-0.23, -1.02], [0.11, -1.04], [-0.19, -0.96], [-0.13, -0.79], [-0.53, -0.32], [-0.78, -0.02], [-0.87, -0.61], [-0.6, -0.32], [-0.06, -0.12],
  [0.1, -0.16], [-0.07, -0.11], [0, 0.03], [0.1, 0.02], [-0.01, -0.15]
]])
    # node_demand = torch.tensor([[0.2413793103448276, 1.0, 0.4482758620689655, 0.6896551724137931, 0.3448275862068966, 0.5862068965517241, 0.41379310344827586, 0.9310344827586207, 0.27586206896551724, 0.27586206896551724, 0.2413793103448276, 0.3793103448275862, 0.20689655172413793, 0.5862068965517241, 0.3448275862068966, 0.41379310344827586, 0.8620689655172413, 0.6551724137931034, 0.20689655172413793, 0.41379310344827586]])

    subset_depot_xy = depot_xy.expand(batch_size, -1, -1)
    subset_node_xy = torch.empty(size=(batch_size, subset_size, 2))
    subset_node_demand = torch.empty(size=(batch_size, subset_size), dtype=torch.float32)

    for i in range(batch_size):
        indices = torch.randperm(problem_size)[:subset_size]
        subset_node_xy[i] = node_xy[0, indices]
        subset_node_demand[i] = node_demand[0, indices]

    return subset_depot_xy, subset_node_xy, subset_node_demand


def calculate_distance_matrix(depot_xy, node_xy):
    """
    Compute the distance matrix for each subinstance
    """
    batch_size = depot_xy.size(0)
    subset_size = node_xy.size(1)

    all_nodes_xy = torch.cat([depot_xy, node_xy], dim=1)  # shape: (batch_size, subset_size + 1, 2)

    diff = all_nodes_xy.unsqueeze(2) - all_nodes_xy.unsqueeze(1)
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1))

    return dist_matrix


def generate_preference_matrix(distance_matrix):
    """
    Generate a preference matrix with normal distribution perturbation
    """
    E = np.random.uniform(low=0.8, high=1.2, size=distance_matrix.shape)
    preference_matrix = distance_matrix * torch.tensor(E)
    return preference_matrix


def nearest_neighbor_heuristic(depot_xy, node_xy, node_demand, truck_capacity, preference_matrix):
    """
    Solve CVRP with heuristic algorithm
    """
    num_nodes = len(node_xy)
    unvisited = set(range(num_nodes))
    route = [0]
    current_node = -1
    capacity_left = truck_capacity
    # routes = []
    # total_distance = 0.0

    while unvisited:
        current_xy = depot_xy if current_node == -1 else node_xy[current_node]
        distances = [(node, preference_matrix[current_node + 1, node + 1]) for node in unvisited]
        distances.sort(key=lambda x: x[1])

        next_node = None
        for node, distance in distances:
            if node_demand[node] <= capacity_left:
                next_node = node
                next_distance = distance
                break

        if next_node is None:
            route.append(0)
            current_node = -1
            capacity_left = truck_capacity
            continue

        route.append(next_node + 1)
        unvisited.remove(next_node)
        capacity_left -= node_demand[next_node]
        # total_distance += next_distance
        current_node = next_node

    if current_node != -1:
        route.append(0)

    return route


def batch_nearest_neighbor_heuristic(depot_xy_batch, node_xy_batch, node_demand_batch, truck_capacity_batch,
                                     preference_matrix_batch):

    batch_size, num_nodes, _ = node_xy_batch.shape

    # 初始化
    routes_batch = torch.zeros((batch_size, num_nodes * 2 + 1), dtype=torch.long)
    route_lengths = torch.ones(batch_size, dtype=torch.long)
    unvisited_mask = torch.ones((batch_size, num_nodes), dtype=torch.bool)
    current_node = torch.full((batch_size,), -1, dtype=torch.long)
    capacity_left = truck_capacity_batch.clone().detach()
    all_done = torch.zeros(batch_size, dtype=torch.bool)

    step = 0

    while not all_done.all():
        current_xy = torch.where(current_node[:, None] == -1, depot_xy_batch,
                                 node_xy_batch[torch.arange(batch_size), current_node])

        dist_to_unvisited = preference_matrix_batch[
            torch.arange(batch_size).unsqueeze(1),
            current_node.unsqueeze(1).clamp(min=0) + 1,
            torch.arange(num_nodes) + 1
        ]

        dist_to_unvisited[torch.arange(batch_size), current_node.clamp(min=0)] = float('inf')

        dist_to_unvisited[~unvisited_mask] = float('inf')

        next_node = torch.argmin(dist_to_unvisited, dim=1)
        next_dist = torch.gather(dist_to_unvisited, 1, next_node.unsqueeze(1)).squeeze(1)
        next_node[all_done] = -1

        valid_node = node_demand_batch[torch.arange(batch_size), next_node] <= capacity_left
        next_node[~valid_node] = -1

        visit_depot = next_node == -1
        routes_batch[torch.arange(batch_size), route_lengths] = torch.where(visit_depot, 0, next_node + 1)
        route_lengths += 1

        not_depot_mask = next_node != -1
        unvisited_mask[torch.arange(batch_size), next_node] = torch.where(not_depot_mask, torch.tensor(0, dtype=torch.bool), unvisited_mask[torch.arange(batch_size), next_node])
        capacity_left -= node_demand_batch[torch.arange(batch_size), next_node] * valid_node

        current_node = torch.where(visit_depot, -1, next_node)

        capacity_left = torch.where(visit_depot, truck_capacity_batch, capacity_left)

        all_done = (~unvisited_mask).all(dim=1)

    routes_batch[torch.arange(batch_size), route_lengths] = 0
    route_lengths += 1

    max_route_length = route_lengths.max().item()

    mask = torch.arange(max_route_length,
                        device=route_lengths.device).expand(batch_size, max_route_length) < route_lengths.unsqueeze(1)

    padded_routes_batch = torch.zeros((batch_size, max_route_length), dtype=torch.long)
    padded_routes_batch[mask] = routes_batch[:, :max_route_length][mask]
    # print('base route:', padded_routes_batch.tolist())

    return padded_routes_batch, routes_batch


def solve_cvrp_and_update_problems(problem_data, num_solutions):
    solutions = []
    for _ in range(num_solutions):
        for problem in problem_data:
            depot_xy = problem['depot_xy']
            node_xy = problem['node_xy']
            node_demand = problem['node_demand']
            truck_capacity = problem['truck_capacity']
            preference_matrix = torch.tensor(problem['preference_matrix'])

            routes = nearest_neighbor_heuristic(depot_xy, node_xy, node_demand, truck_capacity, preference_matrix)

            solutions.append({
                'routes': routes,
                # 'total_distance': float(total_distance),
            })

    return solutions


def solve_cvrp_with_history(depot_xy_batch, node_xy_batch, node_demand_batch, truck_capacity_batch,
                            preference_matrix_batch, num_perturbations=29):

    batch_size = depot_xy_batch.size(0)

    base_solutions, history_no_padding = batch_nearest_neighbor_heuristic(
        depot_xy_batch, node_xy_batch, node_demand_batch, truck_capacity_batch, preference_matrix_batch
    )
    max_route_length = base_solutions.size(1)
    all_solutions_batch = torch.zeros((batch_size, num_perturbations + 1, max_route_length),
                                      dtype=torch.long)

    for i in range(batch_size):
        all_solutions_batch[i, 0] = base_solutions[i]

        for j in range(num_perturbations):
            perturbed_solution = perturb_solution(base_solutions[i])
            all_solutions_batch[i, j + 1] = perturbed_solution

    return all_solutions_batch, history_no_padding


def perturb_solution(solution, perturbation_factor=0.05):

    perturbed_solution = solution.clone().detach()

    non_depot_indices = [i for i in range(1, len(solution) - 1) if solution[i] != 0]

    for i in non_depot_indices:
        if np.random.rand() < perturbation_factor:
            swap_idx = np.random.choice(non_depot_indices)
            perturbed_solution[i], perturbed_solution[swap_idx] = perturbed_solution[swap_idx], perturbed_solution[i]

    return perturbed_solution

def save_problems_to_data_file(num_truck, truck_capacity, depot_xy, node_xy, node_demand, distance_matrix,
                               preference_matrix, filename):
    data_list = []
    for i in range(depot_xy.size(0)):
        problem_data = {
            'num_trucks': num_truck,
            'truck_capacity': truck_capacity,
            'depot_xy': depot_xy[i].tolist(),
            'node_xy': node_xy[i].tolist(),
            'node_demand': node_demand[i].tolist(),
            'distance_matrix': distance_matrix[i].tolist(),
            'preference_matrix': preference_matrix[i].tolist()
        }
        data_list.append(problem_data)
    with open(filename, 'w') as f:
        json.dump(data_list, f, indent=4)


def save_problems_to_pickle(num_truck, truck_capacity, depot_xy, node_xy, node_demand, distance_matrix,
                            preference_matrix, filename):
    data_list = []
    for i in range(depot_xy.size(0)):
        problem_data = {
            'num_trucks': num_truck,
            'truck_capacity': truck_capacity,
            'depot_xy': depot_xy[i],
            'node_xy': node_xy[i],
            'node_demand': node_demand[i],
            'distance_matrix': distance_matrix[i],
            'preference_matrix': preference_matrix[i]
        }
        data_list.append(problem_data)
    with open(filename, 'wb') as f:
        pickle.dump(data_list, f)


def save_solutions_to_data(problem_data, filename):
    with open(filename, 'w') as f:
        json.dump(problem_data, f, indent=4)


def save_solutions_to_pickle(problem_data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(problem_data, f)


if __name__ == '__main__':
    batch_size = 100
    problem_size = 20
    subset_size = 5
    num_truck = 1
    capacity = 1
    problem_filename = 'cvrp5_problem.data'
    depot_xy, node_xy, node_demand = sample_subinstances(batch_size, problem_size, subset_size)
    distance_matrix = calculate_distance_matrix(depot_xy, node_xy)
    preference_matrix = generate_preference_matrix(distance_matrix)
    save_problems_to_data_file(num_truck, capacity, depot_xy, node_xy, node_demand, distance_matrix, preference_matrix,
                               problem_filename)
    with open(problem_filename, 'r') as f:
        problem_data = json.load(f)

    problem_with_history = solve_cvrp_with_history(problem_data)
    save_solutions_to_data(problem_with_history, problem_filename)












