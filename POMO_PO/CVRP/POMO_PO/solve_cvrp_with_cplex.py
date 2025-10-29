import sys
import math
import json
from collections import namedtuple
from docplex.cp.model import CpoModel, CpoParameters
from docplex.cp.model import element, integer_var

import docplex.cp.solver.solver as solver
from docplex.cp.utils import compare_natural


# Ensure the solver version is appropriate
solver_version = solver.get_version_info()['SolverVersion']
if compare_natural(solver_version, '22.1.1.0') < 0:
    print('Warning solver version', solver_version, 'is too old for', __file__)
    exit(0)

TIME_FACTOR = 10

class CVRPProblem:
    def __init__(self):
        self.nb_trucks = -1
        self.truck_capacity = -1
        self.nb_customers = -1
        self.depot_xy = None
        self.customers_xy = []
        self.demands = []
        self._xy = None

    def load_from_data(self, data):
        self.nb_trucks = data['num_trucks']
        self.truck_capacity = data['truck_capacity']
        self.depot_xy = data['depot_xy']
        self.customers_xy = data['node_xy']
        self.demands = data['node_demand']
        self.nb_customers = len(self.customers_xy)
        self._xy = self.depot_xy + self.customers_xy
        # print('node_xy:', self.customers_xy)
        # print('_xy:', self._xy)

    def get_num_nodes(self): return self.nb_customers + 1

    def get_nb_trucks(self): return self.nb_trucks

    def get_capacity(self): return self.truck_capacity

    def get_demand(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return self.demands[i - 1]

    def _get_distance(self, from_, to_):
        c1, c2 = self._xy[from_], self._xy[to_]
        # if isinstance(c1[0], list):
        #     c1 = c1[0]
        # if isinstance(c2[0], list):
        #     c2 = c2[0]
        # print('c1:', c1, 'c2:', c2)
        dx, dy, d = c2[0] - c1[0], c2[1] - c1[1], 0.0
        d = math.sqrt(dx * dx + dy * dy)
        return int(math.floor(d * TIME_FACTOR))

    def get_distance(self, from_, to_):
        assert from_ >= 0
        assert from_ < self.get_num_nodes()
        assert to_ >= 0
        assert to_ < self.get_num_nodes()
        return self._get_distance(from_, to_)

    def get_distance_matrix(self):
        num_nodes = self.get_num_nodes()
        distance_matrix = [
            [self.get_distance(i, j) for j in range(num_nodes)]
            for i in range(num_nodes)
        ]
        return distance_matrix


class VRP:
    VisitData = namedtuple("CustomerData", "demand")
    def __init__(self, pb):
        # Sizes
        self._num_veh = pb.get_nb_trucks()
        self._num_cust = pb.get_num_nodes() - 1
        self._n = self._num_cust + self._num_veh * 2

        # First, last, customer groups
        self._first = tuple(self._num_cust + i for i in range(self._num_veh))
        self._last = tuple(self._num_cust + self._num_veh + i for i in range(self._num_veh))
        self._cust = tuple(range(self._num_cust))

        # Load limits
        self._capacity = pb.get_capacity()

        # Node mapping
        pnode = [i + 1 for i in range(self._num_cust)] + [0] * (2 * self._num_veh)

        # Visit data
        self._visit_data = \
            tuple(VRP.VisitData(pb.get_demand(pnode[c])) for c in self._cust) + \
            tuple(VRP.VisitData(0) for _ in self._first + self._last)

        # Distance
        self._distance = [
          [ pb.get_distance(pnode[i], pnode[j]) for j in range(self._n) ]
          for i in range(self._n)
        ]

    def first(self): return self._first
    def last(self): return self._last
    def vehicles(self): return zip(range(self._num_veh), self._first, self._last)
    def customers(self): return self._cust
    def all(self): return range(self._n)
    def get_num_customers(self): return self._num_cust
    def get_num_visits(self): return self._n
    def get_num_vehicles(self): return self._num_veh
    def get_first(self, veh): return self._first[veh]
    def get_last(self, veh): return self._last[veh]
    def get_capacity(self): return self._capacity
    def get_demand(self, i): return self._visit_data[i].demand
    def get_distance(self, i, j): return self._distance[i][j]


class DataModel:
    vrp = None
    prev = None
    veh = None
    load = None
    params = None


def build_model(cvrp_prob, tlim):
    data = DataModel()
    vrp = VRP(cvrp_prob)
    num_cust = vrp.get_num_customers()
    num_vehicles = vrp.get_num_vehicles()
    n = vrp.get_num_visits()

    mdl = CpoModel()

    # Prev variables, circuit, first/last
    prev = [mdl.integer_var(0, n - 1, "P{}".format(i)) for i in range(n)]
    for v,fv,lv in vrp.vehicles():
        mdl.add(prev[fv] == vrp.get_last((v - 1) % num_vehicles))

    before = vrp.customers() + vrp.first()
    for c in vrp.customers():
        mdl.add(mdl.allowed_assignments(prev[c], before))
        mdl.add(prev[c] != c)

    for _,fv,lv in vrp.vehicles():
        mdl.add(mdl.allowed_assignments(prev[lv], vrp.customers() + (fv,)))

    mdl.add(mdl.sub_circuit(prev))

    # Vehicle
    veh = [mdl.integer_var(0, num_vehicles - 1, "V{}".format(i)) for i in range(n)]
    for v, fv, lv in vrp.vehicles():
        mdl.add(veh[fv] == v)
        mdl.add(veh[lv] == v)
        mdl.add(mdl.element(veh, prev[lv]) == v)
    for c in vrp.customers():
        mdl.add(veh[c] == mdl.element(veh, prev[c]))

    # Demand
    load = [mdl.integer_var(0, vrp.get_capacity(), "L{}".format(i)) for i in range(num_vehicles)]
    used = mdl.integer_var(0, num_vehicles, 'U')
    cust_veh = [veh[c] for c in vrp.customers()]
    demand = [vrp.get_demand(c) for c in vrp.customers()]
    mdl.add(mdl.pack(load, cust_veh, demand, used))

    # Distance
    all_dist = []
    for i in vrp.customers() + vrp.last():
        ldist = [vrp.get_distance(j, i) for j in range(n)]
        all_dist.append(mdl.element(ldist, prev[i]))
    total_distance = mdl.sum(all_dist) / TIME_FACTOR

    # Variables with inferred values
    mdl.add(mdl.inferred(cust_veh + load + [used]))

    # Objective
    mdl.add(mdl.minimize(total_distance))



    # KPIs
    mdl.add_kpi(used, 'Used')

    # Solver params setting
    params = CpoParameters()
    params.SearchType = 'Restart'
    params.LogPeriod = 10000
    if tlim is not None:
        params.TimeLimit = tlim

    mdl.set_parameters(params=params)

    data.vrp = vrp
    data.prev = prev
    data.veh = veh
    data.load = load
    data.params = params

    return mdl, data


# def display_solution(sol, data):
#     vrp = data.vrp
#     sprev = tuple(sol.solution[p] for p in data.prev)
#
#     for v, fv, lv in vrp.vehicles():
#         route = []
#         nd = lv
#         while nd != fv:
#             route.append(nd)
#             nd = sprev[nd]
#         route.append(fv)
#         route.reverse()
#         print('Veh {} --->'.format(v), route, end="")
#         if len(route) >= 2:
#             arrive = 0
#             total_distance = 0
#             total_load = 0
#             for idx, nd in enumerate(route):
#                 if nd != route[-1]:
#                     nxt = route[idx + 1]
#                     locald = vrp.get_distance(nd, nxt)
#                     total_distance += locald
#                     if nd != route[0]:
#                         total_load += data.vrp.get_demand(nd)
#             print('          distance: {} load {}'.format(total_distance, total_load))
#         else:
#             print(' Empty')
#
#     print("Total distance: ", sol.get_objective_values()[0])

def display_solution(sol, data):
    vrp = data.vrp
    sprev = tuple(sol.solution[p] for p in data.prev)

    all_routes = []

    for v, fv, lv in vrp.vehicles():
        route = []
        nd = lv
        while nd != fv:
            route.append(nd)
            nd = sprev[nd]
        route.append(fv)
        route.reverse()

        # 显示路径时将客户节点调整为1到20，起点和终点调整为0
        route_display = [0 if i >= data.vrp.get_num_customers() else i + 1 for i in route]
        print('Veh {} --->'.format(v), route_display, end="")

        if len(route) >= 2:
            total_distance = 0
            total_load = 0
            for idx, nd in enumerate(route):
                if nd != route[-1]:
                    nxt = route[idx + 1]
                    locald = vrp.get_distance(nd, nxt)
                    total_distance += locald
                    if nd != route[0]:
                        total_load += data.vrp.get_demand(nd)
            print('          distance: {} load {}'.format(total_distance, total_load))
        else:
            print(' Empty')

        all_routes.extend(route_display)

    # 去掉重复的0，并拼接所有路径，同时确保起点0存在
    combined_route = [0]
    for node in all_routes:
        if node == 0 and combined_route[-1] == 0:
            continue
        combined_route.append(node)

    print("Combined route: ", combined_route)
    print("Total distance: ", sol.get_objective_values()[0])

    return combined_route, sol.get_objective_values()[0]


def create_cost_matrix(solution, num_customers):
    # 初始化20x20的成本矩阵
    # print('num_customers:', num_customers)
    cost_matrix = [[0 for _ in range(num_customers)] for _ in range(num_customers)]

    # 填充成本矩阵
    for i in range(len(solution) - 1):
        from_node = solution[i]
        to_node = solution[i + 1]
        if from_node != 0 and to_node != 0:
            cost_matrix[from_node - 1][to_node - 1] = 1  # 调整索引为0基
    return cost_matrix


def update_problem_data(problem_data, combined_route, total_distance, distance_matrix):
    problem_data['solution'] = {
        'route': combined_route,
        'total_distance': total_distance,
        'distance_matrix': distance_matrix
    }


def save_problems_to_data_file(filename, problems_data):
    with open(filename, 'w') as f:
        json.dump(problems_data, f, indent=4)


def load_problems_from_data_file(filename):
    with open(filename, 'r') as f:
        data_list = json.load(f)
    return data_list


if __name__ == "__main__":
    filename = 'cvrp5_problem.data'
    tlim = 10
    if len(sys.argv) == 2:
        tlim = float(sys.argv[1])

    problems_data = load_problems_from_data_file(filename)

    for idx, problem_data in enumerate(problems_data):
        print(f"Solving problem {idx + 1}/{len(problems_data)}")
        cvrp_prob = CVRPProblem()
        cvrp_prob.load_from_data(problem_data)
        model, data_model = build_model(cvrp_prob, tlim)
        solution = model.solve()
        if solution:
            combined_route, total_distance = display_solution(solution, data_model)
            distance_matrix = cvrp_prob.get_distance_matrix()
            update_problem_data(problem_data, combined_route, total_distance, distance_matrix)

    save_problems_to_data_file(filename, problems_data)
