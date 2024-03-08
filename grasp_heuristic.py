import random
import gurobi_model
import copy
import time


def time_increase(node, node_before, node_after, distances):
    """
    Potential time increase when adding a stop to the tour

    Parameters:
    node (str): stop
    node_before (str): stop in tour
    node_after (str): stop in tour
    distances (dict(str,str: float)): time distances between stops

    Returns:
    float: time increase when adding a stop at this position to the tour
    """
    return distances[node_before, node] + distances[node, node_after] - distances[node_before, node_after]


def segment(stop):
    """
    Return segment (int) of a stop (str)
    """
    return int(stop[1:3])


def request(node):
    """
    Return request (str) to a request location (str)
    """
    return 'r' + node[1:]


def corresponding_start_or_end(node):
    """
    Return corresponding location (str) of a request location (str)
    """
    if node[0] == 's':
        return 'e' + node[1:]
    if node[0] == 'e':
        return 's' + node[1:]


def delete_stop(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, stop):
    """
    Delete a stop from all dictionaries
    """
    for n in dict_stops[stop]:
        if n in dict_reqs:
            dict_reqs[n].remove(stop)

    del dict_stops[stop]
    if stop in stops_benefit:
        del stops_benefit[stop]
        del stops_dist[stop]
        del stops_greedy_criterion[stop]


def remove_request_nodes(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, stop, factor,
                         demand):
    """
    Delete servable requests and update greedy criterion

    Parameters:
    dict_reqs (dict(request node: list(stops))): all stops in walking distance to a request node
    dict_stops (dict(stop: list(request nodes))): all requests in walking distance to a stop
    stops_benefit (dict(stop: float)): potential benefit of a stop
    stops_dist (dict(stop: float)): potential time increase of a stop
    stops_greedy_criterion (dict(stop: float)): greedy criterion for stops
    stop (str): stop that was newly added to the tour
    factor (float): completion bonus
    demand (dict(request: int)): number of users per request

    Updates greedy criterion and dictionaries
    """
    for n in dict_stops[stop]:
        r = request(n)
        factor2 = 1
        n_corresponding = corresponding_start_or_end(n)

        # completing a request is worth more
        if n_corresponding not in dict_reqs:
            factor2 += factor

        # removing the benefit of this request node from other stops
        for s in dict_reqs[n]:
            if s != stop:
                dict_stops[s].remove(n)
                if s in stops_benefit:
                    stops_benefit[s] -= factor2 * demand[r]
                    stops_greedy_criterion[s] = stops_benefit[s] / stops_dist[s]

        # deleting this request node from the dict
        del dict_reqs[n]

        # increasing the benefit for completing this request
        if n_corresponding in dict_reqs:
            for s in dict_reqs[n_corresponding]:
                if s in stops_benefit:
                    stops_benefit[s] += factor * demand[r]
                    stops_greedy_criterion[s] = stops_benefit[s] / stops_dist[s]

    # delete the stop
    delete_stop(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, stop)


def seg_time_to_departure_times(seg_time, earliest):
    """
    Convert the seg times into departure times at compulsory stops

    Parameters:
    seg_time (list(float)): for each segment, how much time the current segment tour takes the bus
    earliest (list(float)): for each compulsory stop the earliest allowed departure time

    Returns:
    list(float): departure times at compulsory stops
    list(float): bus waiting times at compulsory stops
    """
    # convert segment times to departure times
    tmp = [0]
    waiting = []
    for t in seg_time:
        tmp.append(tmp[-1] + t)
        waiting.append(0)

    # ensure to comply with the earliest time window
    for i in range(len(tmp)):
        if tmp[i] < earliest[i]:
            difference = earliest[i] - tmp[i]
            waiting[i - 1] += difference
            for j in range(i, len(tmp)):
                tmp[j] += difference

    return tmp, waiting


def where_to_wait(instance, tour, current_passengers, departure_times, waiting_times):
    """
    Find the best compulsory stops to wait and return objective value of waiting times for objective value approximation

    Parameters:
    instance (Instance): all model parameters and sets
    tour (list(list(str))): current tour
    current_passengers (dict(str,str: int)): current bus passengers of each arc
    departure_times (list(float)): departure times at compulsory stops
    waiting_times (list(float)): waiting times times at compulsory stops

    Returns:
    float: objective value of  waiting times
    """
    passengers = [100]
    waiting2 = [0] + waiting_times[:]
    space = [a - b for a, b in zip(instance.b, departure_times)]
    for tour_seg in tour:
        passengers.append(current_passengers[tour_seg[-2], tour_seg[-1]])

    left = 0
    for right in range(len(waiting2)):
        w = waiting2[right]
        while w > 0 and left < right:
            tmp = passengers[left:right + 1]
            i = tmp.index(min(tmp))
            left = left + i
            if left == right:
                pass
            else:
                min_space = min(space[left:right])
                if min_space > w:
                    for l2 in range(left, right):
                        space[l2] -= w
                    waiting2[left] += w
                    w = 0
                    waiting2[right] = 0
                else:
                    if min_space <= 0:
                        left = left + 1
                    else:
                        for l2 in range(left, right):
                            space[l2] -= min_space
                        w -= min_space
                        waiting2[left] += min_space
                        waiting2[right] -= min_space

    return sum([a * b for a, b in zip(passengers, waiting2)])


def time_infeasibility(current, earliest, latest, seg, seg_time_increase):
    """
    Check time infeasibility when increasing the time of a segment

    Parameters:
    current list(float): current segment times
    earliest list(float): earliest departure at compulsory stops
    latest list(float): latest departure at compulsory stops
    seg (int): segment
    seg_time_increase (float): potential time increase in that segment

    Returns:
    bool: if True, then this time increase is infeasible
    """
    # time when adding a stop in segment seg
    tmp = current.copy()
    tmp[seg] += seg_time_increase

    # comply with earliest departure time
    tmp, _ = seg_time_to_departure_times(tmp, earliest)

    # comply with latest departure time
    if any(tmp[i] > latest[i] for i in range(len(tmp))):
        return True
    return False


def update_stop_dist(tour, stops_benefit, stops_dist, stops_greedy_criterion, distances, nodes_seg):
    """
    Delete servable requests and update greedy criterion

    Parameters:
    tour (list(list(str))): tour as list of segments, which are lists of stops
    stops_benefit (dict(stop: float)): potential benefit of a stop
    stops_dist (dict(stop: float)): potential time increase of a stop
    stops_greedy_criterion (dict(stop: float)): greedy criterion for stops
    distances (dict(str,str: float): travel times

    Updates potential tour increase as part of the greedy criterion
    """
    for s in nodes_seg:
        ss = str(s)
        # if not yet added to tour or declined from it
        if ss in stops_dist:
            increase = 1000
            # by how much would the tour length increase if we ad stop ss to the tour
            for i in range(len(tour) - 1):
                increase = min(increase, time_increase(ss, tour[i], tour[i + 1], distances))

            # update stops_dist and stops_greedy_criterion
            stops_dist[ss] = max(increase, 0.01)
            stops_greedy_criterion[ss] = stops_benefit[ss] / stops_dist[ss]


def tour_position(stop, distances, seg_tour):
    """
    Find the best position to insert a stop into the tour

    Parameters:
    stop (str): stop to insert
    distances (dict(str,str: float): travel times
    seg_tour (list(str)): current path of a segment

    Returns:
    (list(str)): updated segment tour
    (float): tour time increase
    """
    # find the best spot to insert the stop into the tour
    tour_seg = copy.deepcopy(seg_tour)
    increase = 10000
    position = -1
    for i in range(len(tour_seg) - 1):
        tmp = distances[tour_seg[i], stop] + distances[stop, tour_seg[i + 1]] - distances[tour_seg[i], tour_seg[i + 1]]
        if tmp < increase:
            increase = tmp
            position = i + 1

    tour_seg.insert(position, stop)

    return tour_seg, increase


def add_stop_to_tour(stop, stops_used, stops_not_used, tour, seg, seg_tour, seg_time, seg_time_increase):
    """
    Insert stop into tour

    Parameters:
    stop (str): stop to insert
    stops_used (list(str)): active stops in the tour
    stops_not_used (list(str)): inactive stops not in the tour
    tour (list(list(str))): tour as list of segments, which are lists of stops
    seg (int): segment
    seg_tour (list(str)): current path of segment
    seg_time (list(float)): segment times
    seg_time_increase (float): tour time increase in this segment

    Returns:
    (list(str)): updated segment tour
    (float): tour time increase
    """
    if stop in stops_not_used:
        stops_not_used.remove(stop)
    stops_used.append(stop)
    tour[seg] = seg_tour
    seg_time[seg] += seg_time_increase


def dict_reqs_and_stops(instance):
    """
    Create lookup table that links request nodes and stops with respect to the maximum walking willingness
    """
    # dict that links reachable stops to request nodes
    dict_reqs = {n: [i for i, j in instance.G.in_edges(n) if i not in instance.N_S] +
                    [j for i, j in instance.G.out_edges(n) if j not in instance.N_E]
                 for n in instance.N_S + instance.N_E}

    # dict that links the request nodes to stops in range
    dict_stops = {s: [j for i, j in instance.G.out_edges(s) if j in instance.N_E] +
                     [i for i, j in instance.G.in_edges(s) if i in instance.N_S] for s in instance.N_B}

    return dict_reqs, dict_stops


def find_closest_stop(instance, dict_reqs, stops_used, tour, node):
    """
    Find closest stop to the current tour that can serve the request location

    Parameters:
    instance (Instance): all model parameters and sets
    dict_reqs (dict(request location: list(stops)): stops within range of request location
    stops_used: active stops
    tour (list(list(str))): current tour
    node (str): request location

    Returns:
    (None), if request location already servable
    (str): closest stop to the tour that can serve the request loction
    (float): resulting tour time increase
    """
    # if node is not already reachable
    closest_stop = None
    serving_costs = 10000
    if not any(s in stops_used for s in dict_reqs[node]):
        for stop in dict_reqs[node]:
            seg = segment(stop)
            _, seg_time_increase = tour_position(stop, instance.tau, tour[seg])
            if seg_time_increase < serving_costs:
                closest_stop = stop
                serving_costs = seg_time_increase
        return serving_costs, closest_stop

    else:
        # if tour can already serve the node
        return None


def costs_to_add_requests(instance, requests_not_served, dict_reqs, stops_used, tour):
    """
    Create second greedy criterion that ranks requests based on how time-consuming it is to add the necessary stops
    to the tour

    Parameters:
    instance (Instance): all model parameters and sets
    requests_not_served (list(requests)): not fulfillable requests
    dict_reqs (dict(request location: list(stops)): stops within range of request location
    stops_used: active stops
    tour (list(list(str))): current tour

    Returns:
    (list): list based on greedy criterion that ranks requests and also includes infos about which stops to add
    """
    request_serving_costs = []
    for i in range(len(requests_not_served) // 2):
        node = requests_not_served[i]
        node2 = corresponding_start_or_end(node)

        closest_stop_info = find_closest_stop(instance, dict_reqs, stops_used, tour, node)
        closest_stop_info2 = find_closest_stop(instance, dict_reqs, stops_used, tour, node2)

        if not closest_stop_info and not closest_stop_info2:
            request_serving_costs.append([0, node, False, False])
        elif not closest_stop_info2:
            request_serving_costs.append([closest_stop_info[0], node, closest_stop_info, False])
        elif not closest_stop_info:
            request_serving_costs.append([closest_stop_info2[0], node, closest_stop_info2, False])
        elif closest_stop_info and closest_stop_info2:
            request_serving_costs.append(
                [closest_stop_info[0] + closest_stop_info2[0], node, closest_stop_info, closest_stop_info2])

    sorted_list = sorted(request_serving_costs, key=lambda x: x[0])
    return sorted_list


def result_dict(instance, tour):
    """
    translate tour into a dict

    Parameters:
    instance (Instance): all model parameters and sets
    tour (list(list(str))): current tour

    Returns:
    (dict(stop,stop: int)): 1, if an arc is part of the tour, 0 otherwise
    """
    # resulting route
    x = {(i, j): 0 for i, j in instance.A_B}
    for seg in tour:
        for i in range(len(seg) - 1):
            x[seg[i], seg[i + 1]] = 1

    return x


def next_stop(stop, tour):
    """
    return next stop of the current tour
    """
    seg = segment(stop)
    if seg == len(tour):
        return stop
    for i in range(len(tour[seg])):
        if tour[seg][i] == stop:
            return tour[seg][i + 1]


def check_order(tour, cur, stop):
    """
    check if stop is before cur in the tour
    """
    s1, s2 = stop, cur
    while True:
        s1 = next_stop(s1, tour)
        s2 = next_stop(s2, tour)
        if s1 == cur:
            return True
        if s2 == stop:
            return False


def time_change_stop_request(instance, tour, n, cur, stop):
    """
    time change when we connect request node n to 'stop' instead of cur

    Parameters:
    instance (Instance): all model parameters and sets
    tour (list(list(str))): current tour
    n (str): request node
    cur (str): current stop to serve n
    stop (str): potential alternative stop to serve n

    Return:
    (float): tour time change when we assign the request node to the alternative
    """
    # recursive
    if check_order(tour, cur, stop):
        return time_change_stop_request(instance, tour, n, stop, cur) * -1

    # correct position for calculation
    else:
        demand = instance.d[request(n)]
        if n[0] == 's':
            time_change = (instance.tau[n, stop] - instance.tau[n, cur]) * demand
        else:
            time_change = (instance.tau[stop, n] - instance.tau[cur, n]) * demand

        while cur != stop:
            nxt = next_stop(cur, tour)
            if n[0] == 's':
                time_change -= instance.tau[cur, nxt] * demand
            else:
                time_change += instance.tau[cur, nxt] * demand
            cur = nxt

        return time_change


def update_passengers(stop, n, tour, demand, assigned_to_requests, assigned_to_stop, current_passengers):
    """
    update passengers currently on the bus when the alternative stop serves n

    Parameters:
    stop (str): we reassign node n to this stop
    n (str): request node
    tour (list(list(str))): current tour
    demand (int): users of this request associated with n
    assigned_to_requests (dict(request node: stop)): current assignment
    assigned_to_stop (dict(stop: request node)): current assignment
    current_passengers (dict(stop, stop: int)): current passengers on an arc

    Update current passengers after reassigning node n
    """
    cur = assigned_to_requests[n]
    assigned_to_stop[cur].remove(n)

    if check_order(tour, cur, stop):
        old = cur
        cur = stop
        while cur != old:
            nxt = next_stop(cur, tour)
            if n[0] == 's':
                current_passengers[cur, nxt] += demand
            else:
                current_passengers[cur, nxt] -= demand
            cur = nxt
        assigned_to_requests[n] = stop
        assigned_to_stop[stop] = assigned_to_stop.get(stop, []) + [n]

    else:
        while cur != stop:
            nxt = next_stop(cur, tour)
            if n[0] == 's':
                current_passengers[cur, nxt] -= demand
            else:
                current_passengers[cur, nxt] += demand
            cur = nxt
        assigned_to_requests[n] = stop
        assigned_to_stop[stop] = assigned_to_stop.get(stop, []) + [n]


def remove_stop(old_stop, instance, tour, seg_time, current_passengers, stops_used, stops_not_used):
    """
    remove a stop from the tour and update the corresponding dicts
    """
    seg_old = segment(old_stop)
    for i in range(len(tour[seg_old])):
        if tour[seg_old][i] == old_stop:
            old_before = tour[seg_old][i - 1]
            old_after = tour[seg_old][i + 1]

    stops_used.remove(old_stop)
    stops_not_used.append(old_stop)

    # update tour and seg_time
    seg_time[seg_old] -= time_increase(old_stop, old_before, old_after, instance.tau)
    tmp = current_passengers[old_before, old_stop]
    assert tmp == current_passengers[old_stop, old_after]
    current_passengers[old_before, old_after] = tmp
    del current_passengers[old_before, old_stop]
    del current_passengers[old_stop, old_after]
    tour[seg_old].remove(old_stop)


def remove_edges(inst, x_fixed):
    """
    Remove arc variables that are not part of the fixed tour

    Parameters:
    inst (instance): instance
    x_fixed (dict(str, str: int)): random seed that was used for the instance creation

    Returns:
    instance: modified instance with a fixed tour
    """
    instance = copy.deepcopy(inst)
    # active bus stops
    stops = set()
    # remove bus driving arcs
    for s1, s2 in x_fixed:
        if x_fixed[s1, s2] == 0:
            instance.A_B.remove((s1, s2))
            instance.A.remove((s1, s2))
            instance.G.remove_edge(s1, s2)
            instance.G_b.remove_edge(s1, s2)
        else:
            stops.add(s1)
            stops.add(s2)

    # remove walking arcs
    for i, j in list(instance.G.edges()):
        if i in instance.N_S and j not in stops and j not in instance.N_E:
            instance.G.remove_edge(i, j)
            instance.A.remove((i, j))
        elif j in instance.N_E and i not in stops and i not in instance.N_S:
            instance.G.remove_edge(i, j)
            instance.A.remove((i, j))

    return instance


def greedy_construction(instance, parameters):
    """
    Initial route generation via greedy adaptive insertion heuristic

    Parameters:
    instance (Instance): all model parameters and sets
    parameters (int, float, int): seed, completion bonus, candidate list length

    Returns:
    initial tour with additional tour data
    """
    ###################################################################################################################
    bonus_completion = parameters[1]
    candidate_list_len = parameters[2]
    random.seed(parameters[0])
    ###################################################################################################################

    tour = [[instance.C[i], instance.C[i + 1]] for i in range(len(instance.C) - 1)]
    seg_time = [instance.tau[instance.C[i], instance.C[i + 1]] for i in range(len(instance.C) - 1)]

    # dict that links reachable stops to request nodes
    dict_reqs, dict_stops = dict_reqs_and_stops(instance)

    # dicts for the greedy criterion
    stops_benefit = {s: sum([instance.d[request(n)] for n in dict_stops[s]]) for s in dict_stops if s not in instance.C}
    stops_dist = {s: max(time_increase(s, tour[segment(s)][0], tour[segment(s)][1], instance.tau), 0.01)
                  for s in stops_benefit}
    stops_greedy_criterion = {s: stops_benefit[s] / stops_dist[s] for s in stops_benefit}

    # removing request nodes that are already reachable by the compulsory stops
    for s in instance.C:
        remove_request_nodes(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, s,
                             bonus_completion,
                             instance.d)

    # while there are stops we have not looked at
    stops_not_used = []
    stops_used = [s for s in instance.C]
    while dict_stops:
        # choose one stop to add from the restricted candidate list
        sorted_items = sorted(stops_greedy_criterion.items(), key=lambda item: item[1], reverse=True)
        candidate_list = sorted_items[:candidate_list_len]
        potential_stop, _ = random.choice(candidate_list)
        seg = segment(potential_stop)

        # if stop helps no additional request node
        if stops_greedy_criterion[potential_stop] == 0:
            stops_not_used.append(potential_stop)
            delete_stop(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, potential_stop)

        # stop would help at least one request
        else:
            seg_tour, seg_time_increase = tour_position(potential_stop, instance.tau, tour[seg])

            # if adding the stop is not feasible
            if time_infeasibility(seg_time, instance.a, instance.b, seg, seg_time_increase):
                stops_not_used.append(potential_stop)
                delete_stop(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion, potential_stop)

            # adding the stop is feasible and so we do it
            else:
                add_stop_to_tour(potential_stop, stops_used, stops_not_used, tour, seg, seg_tour, seg_time,
                                 seg_time_increase)
                remove_request_nodes(dict_reqs, dict_stops, stops_benefit, stops_dist, stops_greedy_criterion,
                                     potential_stop,
                                     bonus_completion, instance.d)

                update_stop_dist(tour[seg], stops_benefit, stops_dist, stops_greedy_criterion, instance.tau,
                                 instance.G_h[seg].nodes())

    # resulting route
    x = result_dict(instance, tour)

    tour_data = stops_used, tour, seg_time, stops_not_used
    return x, tour_data


def greedy_improvement_heuristic(instance, seed, tour_data, run_obj_val, run_obj_val_h, run_info,
                                 no_gurobi=False, for_path=False):
    """
    Greedy improvement heuristic

    Parameters:
    instance (Instance): all model parameters and sets
    seed (int): random seed
    tour_data (tuple(...)): stops_used, tour, seg_time, stops_not_used
    run_obj_val (list(float)): objective values from previous iterations
    run_obj_val_h (list(float)): objective value approximations from previous iterations
    run_info (list(list(int))): run info from previous iterations
    no_gurobi (bool): indicates if approximation instead of gurobi should be used at each iteration
    for_path (bool): indicates if the path-based solution approach calls this heuristic

    Returns:
    if for_path: final tour
    else: result data and objective value
    """
    # [removed stops step1, added request step2, removed stops step3, added stops step4, removed stops step4]
    r_a_r_a_r = [0, 0, 0, 0, 0]

    random.seed(seed)
    dict_reqs, dict_stops = dict_reqs_and_stops(instance)
    stops_used, tour, seg_time, stops_not_used = copy.deepcopy(tour_data)

    # request nodes that cannot be reached
    tmp = [n for n in dict_reqs if not any(s in stops_used for s in dict_reqs[n])]
    # orphan nodes
    tmp2 = [corresponding_start_or_end(n) for n in tmp
            if corresponding_start_or_end(n) not in tmp]
    # unserved request locations
    requests_not_served = tmp + tmp2
    requests_not_served.sort()

    random.shuffle(stops_used)

    # ==== Step 1 (Operator 1): Check for each stop if we can remove it from the tour without loosing requests ====
    i = 0
    while i < len(stops_used):
        stop = stops_used[i]
        # only for optional stops
        if stop not in instance.C:
            useless = True
            for node in dict_stops[stop]:
                # if the current tour can serve the request
                if node not in requests_not_served:
                    # if there is no other stop in the current tour that could also serve this request
                    if not any(stop2 in stops_used for stop2 in dict_reqs[node] if stop2 != stop):
                        useless = False

            # remove stop if useless
            if useless:
                stops_used.remove(stop)
                stops_not_used.append(stop)
                i -= 1
                r_a_r_a_r[0] += 1

                # update tour and seg_time
                seg = segment(stop)
                for i2 in range(len(tour[seg])):
                    if tour[seg][i2] == stop:
                        seg_time[seg] -= time_increase(stop, tour[seg][i2 - 1], tour[seg][i2 + 1], instance.tau)
                        tour[seg].pop(i2)
                        break

        i += 1

    # ==== Step 2 (Operator 2): Check if we can now serve additional requests ====
    request_serving_costs = costs_to_add_requests(instance, requests_not_served, dict_reqs, stops_used, tour)
    while request_serving_costs:
        _, node, stop1_info, stop2_info = request_serving_costs.pop(0)

        # we can already serve this request
        if not stop1_info:
            r_a_r_a_r[1] += 1
            requests_not_served.remove(node)
            requests_not_served.remove(corresponding_start_or_end(node))

        # stop(s) need(s) that we need to add
        else:
            cost1, stop1 = stop1_info
            seg1 = segment(stop1)

            # check if adding the stop(s) is feasible
            time_feasible = False

            if not time_infeasibility(seg_time, instance.a, instance.b, seg1, cost1):
                # if we need to add two stops
                if stop2_info:
                    cost2, stop2 = stop2_info
                    seg2 = segment(stop2)
                    seg_time_copy = seg_time.copy()
                    seg_time_copy[seg1] += cost1
                    if not time_infeasibility(seg_time_copy, instance.a, instance.b, seg2, cost2):
                        # add two stops to the tour
                        seg_tour, seg_time_increase = tour_position(stop1, instance.tau, tour[seg1])
                        add_stop_to_tour(stop1, stops_used, stops_not_used, tour, seg1, seg_tour, seg_time,
                                         seg_time_increase)
                        seg_tour, seg_time_increase = tour_position(stop2, instance.tau, tour[seg2])
                        add_stop_to_tour(stop2, stops_used, stops_not_used, tour, seg2, seg_tour, seg_time,
                                         seg_time_increase)
                        time_feasible = True

                # add one stop to tour
                else:
                    seg_tour, seg_time_increase = tour_position(stop1, instance.tau, tour[seg1])
                    add_stop_to_tour(stop1, stops_used, stops_not_used, tour, seg1, seg_tour, seg_time,
                                     seg_time_increase)
                    time_feasible = True

            # the tour can serve the request now
            if time_feasible:
                r_a_r_a_r[1] += 1
                requests_not_served.remove(node)
                requests_not_served.remove(corresponding_start_or_end(node))
                request_serving_costs = costs_to_add_requests(instance, requests_not_served, dict_reqs, stops_used,
                                                              tour)

    # ==== Step 2b: If this was unsuccessful, use the original tour instead of the modified one ====
    if r_a_r_a_r[1] == 0:
        stops_used, tour, seg_time, stops_not_used = tour_data

    # ==== Step 2c: Assigning request nodes to bus stops and remove orphans ====
    current_passengers = {}
    assigned_to_stop = {}
    assigned_to_requests = {}

    # current passengers
    cp = 0

    # adding dummy node so that the last stop is also considered
    tour[-1].append('end_help')

    for seg_tour in tour:
        for i in range(len(seg_tour) - 1):
            stop = seg_tour[i]
            # store amount of passengers currently on the bus
            current_passengers[stop, next_stop(stop, tour)] = cp
            for n in dict_stops[stop]:
                if n not in requests_not_served:
                    demand = instance.d[request(n)]

                    # if not yet assigned
                    if n not in assigned_to_requests:
                        assigned_to_requests[n] = stop
                        assigned_to_stop[stop] = assigned_to_stop.get(stop, []) + [n]
                        if n[0] == 's':
                            cp += demand
                        else:
                            cp -= demand

                    # already assigned, so let's check if this stop would be better
                    else:
                        cur = assigned_to_requests[n]
                        time_change = time_change_stop_request(instance, tour, n, cur, stop)

                        # assign request to the new stop
                        if time_change < 0:
                            update_passengers(stop, n, tour, demand, assigned_to_requests, assigned_to_stop,
                                              current_passengers)

                    # store amount of passengers currently on the bus
                    current_passengers[stop, next_stop(stop, tour)] = cp
    tour[-1].remove('end_help')

    # remove orphan stops where no passengers get on/off the bus
    for old_stop in reversed(stops_used):
        # only for optional stops that are orphans
        if old_stop not in instance.C and (old_stop not in assigned_to_stop or len(assigned_to_stop[old_stop]) == 0):
            remove_stop(old_stop, instance, tour, seg_time, current_passengers, stops_used, stops_not_used)
            r_a_r_a_r[2] += 1

    # ==== Step 3 (Operator 3): Adding stops that decrease travel time of passengers ====
    # Step 3a: calculate greedy criterion
    stops_greedy_criterion3 = {}
    for stop in stops_not_used:
        seg = segment(stop)

        # stop could be useful
        if any(n in assigned_to_requests for n in dict_stops[stop]):
            # calculate the benefit for individual requests that could use this stop
            benefit = 0
            # time cost when adding this stop to the tour
            seg_tour, seg_time_increase = tour_position(stop, instance.tau, tour[seg])
            tour_tmp = copy.deepcopy(tour)
            tour_tmp[seg] = seg_tour

            # check for every request node in reach of this stop
            for n in dict_stops[stop]:
                # request that the tour can serve
                if n in assigned_to_requests:
                    # time change when node would use the new stop
                    time_change = time_change_stop_request(instance, tour_tmp, n, assigned_to_requests[n], stop)

                    # this node would use this stop
                    if time_change < 0:
                        benefit += time_change * -1

            # there is at least one node that would use this node
            if benefit > 0:
                if seg_time_increase < 0.01:
                    seg_time_increase = 0.01
                stops_greedy_criterion3[stop] = benefit / seg_time_increase

    # convert dict to list that represents greedy criterion
    list_of_tuples = [(key, value) for key, value in stops_greedy_criterion3.items()]
    stops_greedy_criterion3 = sorted(list_of_tuples, key=lambda item: item[1], reverse=True)

    # Step 3b: add new stops and remove old stops that become orphans
    for stop, _ in stops_greedy_criterion3:
        seg = segment(stop)
        seg_tour, seg_time_increase = tour_position(stop, instance.tau, tour[seg])

        # check if adding this stop would be possible
        if not time_infeasibility(seg_time, instance.a, instance.b, seg, seg_time_increase):

            tour_tmp = copy.deepcopy(tour)
            tour_tmp[seg] = seg_tour

            # find current amount of passengers on the bus that would need to drive this detour
            for i in range(len(seg_tour)):
                if seg_tour[i] == stop:
                    stop_before = seg_tour[i - 1]
                    stop_after = seg_tour[i + 1]

                    effective_increase = seg_time_increase
                    # check if an increase of seg time here really affects other passengers
                    # or if they would have to wait at the end of the segment regardless
                    if seg_time[seg] < instance.a[seg + 1] - instance.b[seg]:
                        effective_increase = max(0, (seg_time[seg] + seg_time_increase) -
                                                 (instance.a[seg + 1] - instance.b[seg]))

                    benefit = effective_increase * -1 * current_passengers[stop_before, stop_after]

            for n in dict_stops[stop]:
                if n in assigned_to_requests:

                    # time change when request node would use the new stop
                    time_change = time_change_stop_request(instance, tour_tmp, n, assigned_to_requests[n], stop)

                    # this node would use this stop
                    if time_change < 0:
                        benefit += time_change * -1

            # add stop to tour
            if benefit > 0:
                r_a_r_a_r[3] += 1
                add_stop_to_tour(stop, stops_used, stops_not_used, tour, seg, seg_tour, seg_time, seg_time_increase)
                current_passengers[stop_before, stop] = current_passengers[stop_before, stop_after]
                current_passengers[stop, stop_after] = current_passengers[stop_before, stop_after]
                del current_passengers[stop_before, stop_after]

                # assign requests to the new stop and update current_passengers
                for n in dict_stops[stop]:
                    if n in assigned_to_requests:
                        # time change when node would use the new stop
                        old_stop = assigned_to_requests[n]
                        time_change = time_change_stop_request(instance, tour, n, old_stop, stop)

                        if time_change < 0:
                            demand = instance.d[request(n)]
                            update_passengers(stop, n, tour, demand, assigned_to_requests, assigned_to_stop,
                                              current_passengers)

                            # if stop now useless remove it from the tour
                            if len(assigned_to_stop[old_stop]) == 0:
                                # only for optional stops
                                if old_stop not in instance.C:
                                    remove_stop(old_stop, instance, tour, seg_time, current_passengers, stops_used,
                                                stops_not_used)
                                    r_a_r_a_r[4] += 1

    # ==== Step 4: Results ====
    # if grasp runs for arc formulation -> find best solution
    if not for_path:

        # use approximation for objective value instead of calling gurobi each iteration
        if no_gurobi:
            # calc obj val manually
            del current_passengers[instance.C[-1], instance.C[-1]]
            wal = dri = pen = wai = 0
            obj_val_h = 0
            # bus tour
            for key in current_passengers:
                obj_val_h += instance.tau[key] * current_passengers[key]
                dri += instance.tau[key] * current_passengers[key]
            # walking times
            for key, val in assigned_to_requests.items():
                demand = instance.d[request(key)]
                if key[0] == 's':
                    obj_val_h += instance.tau[key, val] * demand
                    wal += instance.tau[key, val] * demand
                else:
                    obj_val_h += instance.tau[val, key] * demand
                    wal += instance.tau[val, key] * demand
            # penalties
            for n in requests_not_served:
                if n[0] == 's':
                    demand = instance.d[request(n)]
                    obj_val_h += instance.tau[n, corresponding_start_or_end(n)] * demand
                    pen += instance.tau[n, corresponding_start_or_end(n)] * demand
            # waiting times
            departure_times, waiting_times = seg_time_to_departure_times(seg_time, instance.a)
            wai = where_to_wait(instance, tour, current_passengers, departure_times, waiting_times)
            obj_val_h += wai

            run_obj_val_h.append(obj_val_h)
            run_info.append(r_a_r_a_r)

            # resulting route
            x = result_dict(instance, tour)
            instance_tmp = remove_edges(instance, x)

            return [instance_tmp, x], obj_val_h

        # use gurobi for exact results
        else:
            # resulting route
            x = result_dict(instance, tour)
            instance_tmp = remove_edges(instance, x)

            result = gurobi_model.optimize_arc(instance_tmp, x_fixed=x)

            # if result:
            obj_val = result[-1][1]

            # store run results
            run_obj_val.append(obj_val)
            run_info.append(r_a_r_a_r)

            return result, obj_val

    else:
        # if only paths relevant and not best solution
        return tour


def grasp(instance, parameters, no_gu=False):
    """
    GRASP-based solution approach with arc-based model

    Parameters:
    instance (Instance): all model parameters and sets
    parameters (float, int, int): completion bonus, candidate list length, iterations
    no_gu (bool): indicates if we use an objective value approximation instead of gurobi

    Returns:
    result data
    """
    runs = parameters[-1]
    run_obj_val = []
    run_obj_val_h = []
    run_info = []
    best_result = None
    best_obj = 10000000

    t = time.time()
    performance = [[], []]
    # several randomized runs where it is random which stop out of the restricted candidate list is picked next
    for seed in range(runs):
        _, tour_data = greedy_construction(instance, [seed] + parameters)
        result, obj_val = greedy_improvement_heuristic(instance, seed, tour_data, run_obj_val, run_obj_val_h, run_info,
                                                       no_gurobi=no_gu)
        if result and obj_val < best_obj:
            performance[0].append(time.time() - t)
            performance[1].append(obj_val)
            best_obj = obj_val
            best_result = result

    print(f'local search removed stops and added requests:')
    print(run_info)
    print('grasp runs optimization solution value:', run_obj_val)
    print('grasp runs heuristic solution value:', run_obj_val_h)

    if no_gu:
        result = gurobi_model.optimize_arc(best_result[0], x_fixed=best_result[1])
        return result

    else:
        return best_result, performance
