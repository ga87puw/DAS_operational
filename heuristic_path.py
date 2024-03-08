import grasp_heuristic
import gurobi_model
import time


def seg_tour_to_parameters(instance, seg, seg_tour):
    """
    Create all path related parameters of this segment tour

    Parameters:
    instance (Instance): all sets and parameters for arc-based model
    seg (int): segment
    seg_tour (list(str)): segment tour as a list of stops

    Update instance and add all path related parameters of this segment tour
    """
    if seg_tour not in instance.P[seg].values():
        name = f'p_{seg}_{len(instance.P[seg])}'
        instance.P[seg][name] = seg_tour
        instance.I[name] = len(seg_tour) - 1

        # if auxiliary segment
        if seg >= instance.n - 1:
            instance.I[name] = 1
            instance.tau_p[name] = 0
            instance.tau_pi[name] = [0]

        # normal segment
        else:
            t = [0]
            for i in range(len(seg_tour) - 1):
                t.append(t[-1] + instance.tau[seg_tour[i], seg_tour[i + 1]])
            instance.tau_p[name] = t[-1]
            instance.tau_pi[name] = t[:-1]

        # remove second compulsory stop per segment
        path2 = seg_tour[:-1]
        # except for auxiliary segment
        if len(seg_tour) == 1:
            path2 = seg_tour[:]

        for s in instance.N_S:
            tmp = [j for _, j in instance.G.out_edges(s)]
            if any(stop in path2 for stop in tmp):
                instance.theta[s, seg, name] = 1
            else:
                instance.theta[s, seg, name] = 0
        for e in instance.N_E:
            tmp = [i for i, _ in instance.G.in_edges(e)]
            if any(stop in path2 for stop in tmp):
                instance.theta[e, seg, name] = 1
            else:
                instance.theta[e, seg, name] = 0


def full_tour_to_parameters(instance, tour):
    """
    Convert the fixed final tour into segment paths that we add to the instance

    Parameters:
    instance (Instance): all sets and parameters for arc-based model
    tour (list(list(str))): final tour of one iteration

    Update instance with all path related parameters
    """
    for i in range(len(tour)):
        seg_tour_to_parameters(instance, i, tour[i])


def initial_path_generation(instance):
    """
    Create basic paths that only consist of compulsory stops
    """
    # basic paths of just compulsory stops
    for seg in range(instance.n - 1):
        path = [instance.C[seg], instance.C[seg + 1]]
        seg_tour_to_parameters(instance, seg, path)
    seg_tour_to_parameters(instance, instance.n - 1, [instance.C[-1]])


def print_path_number(instance):
    """
    print path infos

    Return:
    (int): total number of paths
    """
    n_paths = []
    for segment_paths in instance.P:
        n_paths.append(len(segment_paths))
    print('paths per segment', n_paths)
    print('total paths,', sum(n_paths))
    return sum(n_paths)


# use grasp to generate multiple paths
def grasp_for_path_generation(instance, parameters, perf=False):
    """
    GRASP-based solution approach with path-based model

    Parameters:
    instance (Instance): all model parameters and sets for arc-based model
    parameters (float, int, int): completion bonus, candidate list length, iterations
    perf (bool): indicates if we use gurobi callbacks to get infos about bound convergence

    Returns:
    result data
    (float): runtime of heuristic without gurobi runtime
    """
    initial_path_generation(instance)
    runs = parameters[-1]

    # multiple runs of the arc heuristic to generate paths
    for seed in range(runs):
        _, tour_data = grasp_heuristic.greedy_construction(instance, [seed] + parameters)
        tour = grasp_heuristic.greedy_improvement_heuristic(instance, seed, tour_data, _, _, _, for_path=True)
        full_tour_to_parameters(instance, tour)

    print_path_number(instance)

    t = time.time()
    # optimize path model
    result_data = gurobi_model.optimize_path(instance, perf=perf)

    return result_data, t


