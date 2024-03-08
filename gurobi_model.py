import gurobipy as gp
import time
import networkx as nx


class Instance:
    """
    A class representing an instance that has all the relevant sets and parameters to model DAS.
    Notation is the same as in the theis.
    """

    def __init__(self, G, G_b, G_c, G_s, G_e, G_h, demand, requests, a_time_window, b_time_window, big_M):
        # graphs
        self.G = G
        self.G_b = G_b
        self.G_h = G_h

        # parameters for arc based formulation
        self.N = sorted(G.nodes())
        self.N_B = sorted(G_b.nodes())
        self.C = sorted(G_c.nodes())
        self.N_S = sorted(G_s.nodes())
        self.S = {r: s for r, s in zip(sorted(requests), self.N_S)}
        self.N_E = sorted(G_e.nodes())
        self.E = {r: e for r, e in zip(sorted(requests), self.N_E)}
        self.A = sorted(G.edges())
        self.A_B = sorted(G_b.edges())
        self.A_h = [x.edges() for x in G_h]
        self.d = demand
        self.R = requests
        self.tau = nx.get_edge_attributes(G, 'weight')
        self.a = a_time_window
        self.b = b_time_window
        self.M = big_M

        # additional parameters for path based formulation
        self.P = [{} for _ in range(len(self.C))]
        self.I = {}
        self.tau_p = {}
        self.tau_pi = {}
        self.theta = {}
        self.n = len(self.C)


class Result_data:
    """
    A class storing result data after an optimization run
    """

    def __init__(self):
        # computational analysis
        self.x_values = None  # tour arcs information
        self.f_values = None  # walking arcs information
        self.denied = None  # rejected requests
        self.penalty = None  # penalty costs
        self.tour_time = None  # tour time in seconds
        self.o_v_parts = None  # objective value composition
        self.model_data = None  # performance data (gurobi callback)
        self.results = None  # runtime, objective value, best bound, gap

        # case study
        # user travel time data
        self.user_walking = {}
        self.user_onboard = {}
        self.user_penalty = {}

        # bus data
        self.path_duration = []  # duration of selected paths
        self.departure_time = []  # departure times at compulsory stops
        self.waiting_bus = [0]  # bus waiting times before compulsory stops


def performance_cb(model, where):
    """
    Gurobi callback to save performance data to model
    """
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)

        # Did objective value or best bound change?
        if not model._data or model._data[-1][1] != cur_obj or model._data[-1][2] != cur_bd:
            model._data.append([runtime, cur_obj, cur_bd])


# get request of a start or end node
def get_request(node):
    """
    Get request of an origin or destination

    Parameters:
    node (str): origin or destination

    Returns:
    str: request
    """
    return f'r{node[1:]}'


def optimize_arc(instance, timelimit=1800, x_fixed=None, op_costs=True, perf=False):
    """
    Solve arc-based model

    Parameters:
    instance (Instance): operational DAS parameters and sets
    timelimit (int): time limit for gurobi.optimize
    x_fixed (dict(str, str: int)): represents fixed tour if there is a fixed tour
    op_costs (bool): if True, consider additional operator cost constraints
    perf (bool): if True, save performance data with callbacks

    Returns:
    different result information
    """
    print('op costs:', op_costs)
    build_time = time.time()
    model = gp.Model()
    model.params.SoftMemLimit = 28

    # Create variables
    # bus route
    if not x_fixed:
        x = {}
        for i, j in instance.A_B:
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x[{i},{j}]')
    else:
        # x is now a parameter instead of a variable
        x = x_fixed

    # departure time
    t = {}
    for j in instance.N_B:
        t[j] = model.addVar(lb=0, name=f't[{j}]')

    # flow
    f = {}
    for i, j in instance.A:
        for r in instance.R:
            f[r, i, j] = model.addVar(vtype=gp.GRB.BINARY, obj=instance.d[r] * instance.tau[i, j],
                                      name=f'f[{r},{i},{j}]')

    # waiting time
    w = {}
    for j in instance.C:
        for i, _ in instance.G_b.in_edges(j):
            for r in instance.R:
                w[r, i, j] = model.addVar(lb=0, obj=instance.d[r], name=f'w[{r},{i},{j}]')

    # Create constraints
    # we can omit the following constraints if the tour is fixed
    if not x_fixed:
        # one bus leaves at the first stop
        model.addConstr(gp.quicksum(x[i, j] for i, j in instance.G_b.out_edges(instance.C[0])) == 1,
                        name='one bus leaves at the first stop')

        # compulsory stops need to be visited
        for h in range(1, len(instance.C)):
            model.addConstr(gp.quicksum(x[i, j] for i, j in instance.G_b.in_edges(instance.C[h])) == 1,
                            name=f'compulsory stop {instance.C[h]} needs to be visited')

        # all stops can only be visited once
        for j2 in instance.N_B:
            if j2 not in instance.C:
                model.addConstr(gp.quicksum(x[i, j] for i, j in instance.G_b.in_edges(j2)) <= 1,
                                name='all stops can only be visited once')

        # stops that are visited must be left afterwards
        for j2 in instance.N_B:
            if j2 != instance.C[0] and j2 != instance.C[-1]:
                model.addConstr(gp.quicksum(x[i, j] for i, j in instance.G_b.in_edges(j2)) ==
                                gp.quicksum(x[j, k] for j, k in instance.G_b.out_edges(j2)),
                                name=f'stop {j2}, if visited, must be left afterwards')

    # only drive to optional stops where at least one passenger gets on/ off the bus
    if op_costs:
        for j2 in instance.N_B:
            if j2 not in instance.C:
                model.addConstr(gp.quicksum(x[i, j] for i, j in instance.G_b.in_edges(j2)) <=
                                gp.quicksum(f[r, i, j] for r, i, j in f if j == j2 and i in instance.N_S) +
                                gp.quicksum(f[r, j, k] for r, j, k in f if j == j2 and k in instance.N_E),
                                name=f'op costs for stop {j2}')

    # only passenger flow between two bus stops if bus drives there
    for i, j in instance.A_B:
        for r in instance.R:
            model.addConstr(f[r, i, j] <= x[i, j], name='only passenger flow between two bus stops if bus drives there')

    # passenger (multi-commodity) flow
    for j2 in instance.N:
        for r in instance.R:
            rhs = 0
            if j2 == instance.S[r]:
                rhs = -1
            elif j2 == instance.E[r]:
                rhs = 1
            model.addConstr(gp.quicksum(f[r, i, j] for i, j in instance.G.in_edges(j2)) -
                            gp.quicksum(f[r, j, k] for j, k in instance.G.out_edges(j2)) == rhs,
                            name='passenger (multi-commodity) flow')

    # earliest time bus arrives at a stop
    for i, j in instance.A_B:
        model.addConstr(t[j] + instance.M * (1 - x[i, j]) >= t[i] + instance.tau[i, j],
                        name=f'earliest time bus arrives at stop {j}')

    # latest time bus arrives at a stop
    for i, j in instance.A_B:
        if j in instance.C:
            for r in instance.R:
                model.addConstr(t[j] - instance.M * (1 - f[r, i, j]) <= t[i] + instance.tau[i, j] + w[r, i, j],
                                name=f'latest time bus arrives at compulsory stop {j}')
        else:
            model.addConstr(t[j] - instance.M * (1 - x[i, j]) <= t[i] + instance.tau[i, j],
                            name=f'latest time bus arrives at optional stop {j}')

    # time windows at compulsory stops
    for h in range(len(instance.C)):
        model.addConstr(instance.a[h] <= t[instance.C[h]], name=f'time window a at compulsory stop {instance.C[h]}')
        model.addConstr(t[instance.C[h]] <= instance.b[h], name=f'time window b at compulsory stop {instance.C[h]}')

    model.params.TimeLimit = timelimit
    model._data = []
    print(f'It took {time.time() - build_time} seconds to build the model')

    # optimize
    if perf:
        model.optimize(callback=performance_cb)
    else:
        model.optimize()

    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.MEM_LIMIT):
        print('Model is infeasible!')
        print(model.Status)
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print(c)
        return False

    else:
        if model.Status == gp.GRB.Status.TIME_LIMIT:
            print('Time limit was reached. Solution might not be optimal!')
            model._data.append([model.Runtime, model.objVal, model.ObjBound])

        if model.Status == gp.GRB.MEM_LIMIT:
            print('Mem limit was reached. Solution might not be optimal!')
            model._data.append([model.Runtime, model.objVal, model.ObjBound])

        # calculate and store tour information
        # route of the bus
        x_values = []
        tour_time = 0

        if not x_fixed:
            for i, j in instance.A_B:
                if abs(1.0 - x[i, j].X) <= model.params.IntFeasTol:
                    x_values.append((i, j))
                    tour_time += instance.tau[i, j]
        else:
            for i, j in instance.A_B:
                if x[i, j] == 1:
                    x_values.append((i, j))
                    tour_time += instance.tau[i, j]

        # used bus stops for customer requests and current passenger
        f_values = [[] for _ in range(3)]
        denied = 0
        penalty, walking, driving, waiting = 0, 0, 0, 0
        passenger = []

        for i, j in instance.A:
            c_p = 0
            for r in instance.R:
                if abs(1.0 - f[r, i, j].X) <= model.params.IntFeasTol:
                    if i in instance.N_S and j in instance.N_E:
                        denied += instance.d[r]
                        penalty += instance.d[r] * instance.tau[i, j]
                        f_values[2].append((i, j))
                    elif i in instance.N_S:
                        walking += instance.d[r] * instance.tau[i, j]
                        f_values[0].append((i, j))
                    elif j in instance.N_E:
                        walking += instance.d[r] * instance.tau[i, j]
                        f_values[1].append((i, j))
                    else:
                        driving += instance.d[r] * instance.tau[i, j]
                        c_p += instance.d[r]

            if not x_fixed:
                if (i, j) in instance.A_B and abs(1.0 - x[i, j].X) <= model.params.IntFeasTol:
                    passenger.append(c_p)
            else:
                if (i, j) in instance.A_B and x[i, j] == 1:
                    passenger.append(c_p)

        # waiting times
        waiting = [0, 0]
        for j in instance.C:
            wait = False
            for i, _ in instance.G_b.in_edges(j):
                for r in instance.R:
                    if w[r, i, j].X > 0:
                        waiting[1] += instance.d[r] * w[r, i, j].X
                        if not wait:
                            waiting[0] += w[r, i, j].X
                            wait = True

        results = model.Runtime, model.objVal, model.ObjBound, model.MIPGap
        o_v_parts = walking, driving, penalty, waiting[1]

        return [x_values, f_values, denied, penalty, passenger, tour_time, waiting, o_v_parts, model._data, results]


def optimize_path(instance, op_costs=True, perf=False, case_study=False, timelimit=240):
    """
        Solve path-based model

        Parameters:
        instance (Instance): operational DAS parameters and sets
        timelimit (int): time limit for gurobi.optimize
        op_costs (bool): if True, consider additional operator cost constraints
        perf (bool): if True, save performance data with callbacks
        case_study (bool): indicate if optimization call comes from case study; then we store additonal information

        Returns:
        different result information
        """
    build_time = time.time()
    model = gp.Model()
    model.params.SoftMemLimit = 28
    # model.params.MIPGap = 0.01

    # Create helping dicts
    s_reachable = {stop: [s for s, _ in instance.G.in_edges(stop) if s in instance.N_S] for stop in instance.N_B}
    e_reachable = {stop: [e for _, e in instance.G.out_edges(stop) if e in instance.N_E] for stop in instance.N_B}

    # Create variables
    # walks
    q_s = {}
    q_e = {}
    for h in range(instance.n):
        for p in instance.P[h]:
            for i in range(instance.I[p]):
                stop = instance.P[h][p][i]
                for s in s_reachable[stop]:
                    q_s[s, h, p, i] = model.addVar(vtype=gp.GRB.BINARY, name=f'f[{s},{h},{p},{i}]')
                for e in e_reachable[stop]:
                    q_e[e, h, p, i] = model.addVar(vtype=gp.GRB.BINARY, name=f'f[{e},{h},{p},{i}]')

    # departure time at compulsory stops
    t = {}
    for h in range(instance.n):
        t[h] = model.addVar(lb=0, name=f't[{h}]')

    # Bus boarding and alighting time
    u = {}
    v = {}
    for r in instance.R:
        u[r] = model.addVar(lb=0, name=f'u[{r}]')
        v[r] = model.addVar(lb=0, name=f'v[{r}]')

    # request satisfaction
    y = {}
    for r in instance.R:
        y[r] = model.addVar(vtype=gp.GRB.BINARY, name=f'y[{r}]')

    # path
    z = {}
    for h in range(instance.n):
        for p in instance.P[h]:
            z[h, p] = model.addVar(vtype=gp.GRB.BINARY, name=f'z[{h},{p}]')

    # set objective
    # gp.quicksum(instance.d_B * instance.tau_p[p] * z[h, p] for h, p in z) +
    model.setObjective(
        gp.quicksum(instance.d[get_request(s)] * instance.tau[s, instance.P[h][p][i]] * q_s[s, h, p, i]
                    for s, h, p, i in q_s) +
        gp.quicksum(instance.d[get_request(e)] * instance.tau[instance.P[h][p][i], e] * q_e[e, h, p, i]
                    for e, h, p, i in q_e) +
        gp.quicksum(-u[r] * instance.d[r] for r in instance.R) +
        gp.quicksum(v[r] * instance.d[r] for r in instance.R) +
        gp.quicksum(instance.d[r] * (1 - y[r]) * instance.tau[instance.S[r], instance.E[r]] for r in instance.R),
        sense=gp.GRB.MINIMIZE
    )

    # Create constraints
    # request satisfaction only if boarding and alighting possible
    for r in instance.R:
        model.addConstr(
            y[r] <= gp.quicksum(instance.theta[instance.S[r], h, p] for h in range(instance.n) for p in instance.P[h]),
            name='request satisfaction start')
        model.addConstr(
            y[r] <= gp.quicksum(instance.theta[instance.E[r], h, p] for h in range(instance.n) for p in instance.P[h]),
            name='request satisfaction end')

    # one path per segment
    for h in range(instance.n):
        model.addConstr(gp.quicksum(z[h, p] for p in instance.P[h]) == 1, name=f'one path for segment {h}')

    # only walk from/to used paths
    for s, h, p, i in q_s:
        model.addConstr(q_s[s, h, p, i] <= z[h, p], name='only walk to used paths')
    for e, h, p, i in q_e:
        model.addConstr(q_e[e, h, p, i] <= z[h, p], name='only walk from used paths')

    # only drive to optional stops where at least one passenger gets on/ off the bus
    if op_costs:
        for h in range(instance.n):
            for p in instance.P[h]:
                for i in range(1, instance.I[p]):
                    model.addConstr(z[h, p] <= gp.quicksum(
                        q_s[s2, h2, p2, i2] for s2, h2, p2, i2 in q_s if h == h2 and p == p2 and i == i2) + gp.quicksum(
                        q_e[e2, h2, p2, i2] for e2, h2, p2, i2 in q_e if h == h2 and p == p2 and i == i2),
                                    name='op costs')

    # walk if and only if request can be fulfilled
    for r in instance.R:
        s = instance.S[r]
        e = instance.E[r]
        model.addConstr(gp.quicksum(q_s[s2, h, p, i] for s2, h, p, i in q_s if s == s2) == y[r],
                        name='walk once start')
        model.addConstr(gp.quicksum(q_e[e2, h, p, i] for e2, h, p, i in q_e if e == e2) == y[r],
                        name='walk once end')

    # departure time at compulsory stops
    for h in range(instance.n - 1):
        model.addConstr(t[h + 1] >= t[h] + gp.quicksum(instance.tau_p[p] * z[h, p] for p in instance.P[h]),
                        name='departure times')

    # time windows at compulsory stops
    for h in range(instance.n):
        model.addConstr(instance.a[h] <= t[h], name=f'time window a at compulsory stop {instance.C[h]}')
        model.addConstr(t[h] <= instance.b[h], name=f'time window b at compulsory stop {instance.C[h]}')

    # boarding and alighting times
    for s, h, p, i in q_s:
        model.addConstr(u[get_request(s)] - instance.M * (1 - q_s[s, h, p, i]) <= t[h] + instance.tau_pi[p][i],
                        name='boarding')
    for e, h, p, i in q_e:
        model.addConstr(v[get_request(e)] + instance.M * (1 - q_e[e, h, p, i]) >= t[h] + instance.tau_pi[p][i],
                        name='alighting')

    # only boarding and alighting times if request served
    for r in instance.R:
        model.addConstr(u[r] - instance.M * y[r] <= 0, name='no boarding')
        model.addConstr(v[r] + instance.M * y[r] >= 0, name='no alighting')

    model.params.TimeLimit = timelimit
    model._data = []
    print(f'It took {time.time() - build_time} seconds to build the model')

    # optimize
    if perf:
        model.optimize(callback=performance_cb)
    else:
        model.optimize()

    result_data = Result_data()

    print('Model status:', model.Status)
    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.MEM_LIMIT):
        print('Model is infeasible!')
        print(model.Status)
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print(c)
        return False

    else:
        if model.Status == gp.GRB.Status.TIME_LIMIT:
            print('Time limit was reached. Solution might not be optimal!')
            model._data.append([model.Runtime, model.objVal, model.ObjBound])

        if model.Status == gp.GRB.MEM_LIMIT:
            print('Mem limit was reached. Solution might not be optimal!')
            model._data.append([model.Runtime, model.objVal, model.ObjBound])


        # calculate interesting result data
        # route of the bus
        x_values = []
        tour_time = 0
        paths_used = []
        path_of_paths = []

        for h in range(instance.n):
            for p in instance.P[h]:
                if abs(1.0 - z[h, p].X) <= model.params.IntFeasTol:
                    result_data.path_duration.append(instance.tau_p[p])
                    paths_used.append(p)
                    path_of_paths.append(instance.P[h][p])
                    for i in range(len(instance.P[h][p]) - 1):
                        x_values.append((instance.P[h][p][i], instance.P[h][p][i + 1]))
                    tour_time += instance.tau_p[p]

        # departure times and waiting times
        for h in range(instance.n):
            result_data.departure_time.append(t[h].X)
            if h > 0:
                result_data.waiting_bus.append(t[h].X - t[h - 1].X - result_data.path_duration[h - 1])

        # other interesting data
        f_values = [[] for _ in range(3)]
        denied = 0
        penalty, walking, driving_waiting = 0, 0, 0

        for r in instance.R:
            s = instance.S[r]
            e = instance.E[r]
            if abs(1 - y[r].X) <= model.params.IntFeasTol:
                driving_waiting += instance.d[r] * (v[r].X - u[r].X)
                result_data.user_onboard[r] = (instance.d[r] * (v[r].X - u[r].X))

                for h in range(instance.n):
                    for p in instance.P[h]:
                        if abs(1 - z[h, p].X) <= model.params.IntFeasTol:
                            for i in range(instance.I[p]):
                                stop = instance.P[h][p][i]
                                if (s, h, p, i) in q_s:
                                    if abs(1.0 - q_s[s, h, p, i].X) <= model.params.IntFeasTol:
                                        walking += instance.d[r] * instance.tau[s, stop]
                                        result_data.user_walking[r] = instance.d[r] * instance.tau[s, stop]
                                        f_values[0].append((s, stop))

                                if (e, h, p, i) in q_e:
                                    if abs(1.0 - q_e[e, h, p, i].X) <= model.params.IntFeasTol:
                                        walking += instance.d[r] * instance.tau[stop, e]
                                        result_data.user_walking[r] += instance.d[r] * instance.tau[stop, e]
                                        f_values[1].append((stop, e))

            else:
                denied += instance.d[r]
                penalty += instance.d[r] * instance.tau[s, e]
                result_data.user_penalty[r] = instance.d[r] * instance.tau[s, e]
                f_values[2].append((s, e))

        results = model.Runtime, model.objVal, model.ObjBound, model.MIPGap
        o_v_parts = walking, driving_waiting, penalty

        res_data = [x_values, f_values, denied, penalty, [-1], tour_time, [-1, -1], o_v_parts, model._data, results]

        if not case_study:
            return res_data
        else:
            result_data.x_values = x_values
            result_data.f_values = f_values
            result_data.denied = denied
            result_data.penalty = penalty
            result_data.tour_time = tour_time
            result_data.o_v_parts = o_v_parts
            result_data.model_data = model._data
            result_data.results = results

            # probably better to return a single object instead of a tuple of different objects as before
            return result_data
