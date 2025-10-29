"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os.path
import numpy as np
from scipy.spatial.distance import pdist, squareform
#from concorde.tsp import TSPSolver

import torch
from tqdm import tqdm

SCALE = 1e6


def _get_travel_distance_2(problems, solution):
    gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

    seq_expanded = problems

    ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

    segment_lengths = ((ordered_seq - rolled_seq) ** 2)

    segment_lengths = segment_lengths.sum(2).sqrt()

    travel_distances = segment_lengths.sum(1)

    return travel_distances

def load_raw_data( episode, begin_index=0):
    print('load raw dataset begin!')

    raw_data_nodes = []
    raw_data_tours = []
    for line in tqdm(open('tsp_data_1000_1.txt', "r").readlines()[0 + begin_index:episode + begin_index], ascii=True):
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

        raw_data_nodes.append(nodes)
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

        raw_data_tours.append(tour_nodes)

    raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
    raw_data_tours = torch.tensor(raw_data_tours, requires_grad=False)
    print(f'load raw dataset done!', )
    return raw_data_nodes,raw_data_tours



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="generate and solve TSP")
    # parser.add_argument("--num_instances", type=int, default=10, help="Numbers of TSP instances")
    # parser.add_argument("--num_nodes", type=int, default=100, help="Numbers of nodes")
    # parser.add_argument("--output_filename", type=str, required=True, help="Output directory")
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--reorder", dest="reorder", action="store_true",
    #                     help="Reorder nodes/tours. training dataset MUST BE reordered")

    # seed=123
    # num_instances=128
    # num_nodes=100
    # reorder=False
    output_filename='tsp_data_1000a'

    #args = parser.parse_args()

    raw_data_nodes, raw_data_tours=load_raw_data(episode=128)


    all_instance_coords = raw_data_nodes.numpy()



    coords = list()



    for instance_coords in all_instance_coords:
        # solver = TSPSolver.from_data(instance_coords[:, 0] * SCALE, instance_coords[:, 1] * SCALE, norm="EUC_2D")
        # solution = solver.solve()
        # solution_closed_tour = list(solution[0]) + [0]

        instance_coords = instance_coords.tolist()
        instance_coords.append(instance_coords[0])

        # compute tour length


        coords.append(instance_coords)

    tour_lens = _get_travel_distance_2(raw_data_nodes, raw_data_tours).tolist()


    np.savez_compressed(os.path.join(output_filename), coords=np.array(coords), tour_lens=tour_lens,
                            reorder=False)

    print("Data transformed and saved to " + output_filename)
