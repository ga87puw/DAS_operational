import json
import gurobi_model
import heuristic_path
import networkx as nx
from datetime import datetime
import os
import pickle
import time
from geopy.distance import geodesic
import folium
from folium.plugins import BeautifyIcon
import webbrowser
import random
import math


def load_json(filename):
    """
    Load  json file

    Parameters:
    filename (str): json file

    Returns:
    (dict): dictionary of json data
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {filename}.")
        return None


def time_to_sec(time_str):
    """
    Convert time object of format "%H:%M:%S" into seconds

    Parameters:
    time_str (str): time as string

    Returns:
    (int): time in seconds
    """
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


def stops_in_walking_distance(n, pos_stops, max_walk):
    """
    Find all stops within the maximum walking willingness of the request location

    Parameters:
    n (float, float): latitude and longitude of request location
    pos_stops (dict(str: float, float)): latitude and longitude of stops
    max_walk (float): maximum walking willingness in meters

    Returns:
    (list(str)): list of stops
    """
    t = []

    min_dist = 1000000
    for k, v in pos_stops.items():
        dist = geodesic(n, v).meters
        min_dist = min(dist, min_dist)
        if dist <= max_walk:
            t.append(k)

    # print problem that no stop is in walking distance
    if len(t) == 0:
        print(f'Request node with coordinates {n} closest stop is {min_dist} meters away')

    return t


# change the request locations, while keeping sure that the request remains viable
def create_new_viable_request(ori, dest, pos_stops, max_walk):
    """
    Change the request locations to represent a lower demand density, while keeping sure that the request remains viable

    Parameters:
    ori (float, float): latitude and longitude of origin
    dest (float, float): latitude and longitude of destination
    pos_stops (dict(str: float, float)): latitude and longitude of stops
    max_walk (float): maximum walking willingness in meters

    Returns:
    (float, float): latitude and longitude of new origin
    (float, float): latitude and longitude of new destination
    """
    lat_plus_max = 0.009  # 1000 m
    lon_plus_max = 0.0135  # 1000 m

    # until the new request is viable
    while True:
        # create new random location
        lat_plus1 = random.uniform(-lat_plus_max, lat_plus_max)
        lon_plus1 = random.uniform(-lon_plus_max, lon_plus_max)
        lat_plus2 = random.uniform(-lat_plus_max, lat_plus_max)
        lon_plus2 = random.uniform(-lon_plus_max, lon_plus_max)
        new_ori = (ori[0] + lat_plus1, ori[1] + lon_plus1)
        new_dest = (dest[0] + lat_plus2, dest[1] + lon_plus2)

        # check if within a 1000-meter radius of original location
        if geodesic(ori, new_ori).meters <= 1000 and geodesic(dest, new_dest).meters <= 1000:

            # check if oriented in the right direction
            if new_ori[0] > new_dest[0] and new_ori[1] < new_dest[1]:

                # check if within service area by having at least two stops in reach
                if len(stops_in_walking_distance(new_ori, pos_stops, max_walk)) > 1:
                    if len(stops_in_walking_distance(new_dest, pos_stops, max_walk)) > 2:

                        # check for min distance between origin and destination
                        if geodesic(new_ori, new_dest).meters > 1500:
                            # return new request locations
                            return new_ori, new_dest


def generate_demand(data, demand_decrease, pos_stops, max_walk):
    """
    Create new request data and save it

    Parameters:
    data (dict()): json file data
    demand_decrease (float): proportion of requests of the high demand scenario that remain
    pos_stops (dict(str: float, float)): latitude and longitude of stops
    max_walk (float): maximum walking willingness in meters

    Generate low-demand scenario request data and save it as pickle file
    """
    request_data = []
    n_req = int(len(data['sampled_building_pairs']) / 2)
    numbers = list(range(n_req))
    numbers.remove(9)  # problems with this request
    random.shuffle(numbers)
    change = math.floor(n_req * demand_decrease)
    numbers = numbers[:change]

    for i in numbers:
        # request
        entry = []
        entry.append(f'r{i:02d}')
        print(f'r{i:02d}')

        name1 = f's{i:02d}'
        name2 = f'e{i:02d}'

        i = i * 2

        # origin
        location1 = (data['sampled_building_pairs'][i]['lat'], data['sampled_building_pairs'][i]['lon'])
        # destination
        location2 = (data['sampled_building_pairs'][i + 1]['lat'], data['sampled_building_pairs'][i + 1]['lon'])

        # alter locations
        location1, location2 = create_new_viable_request(location1, location2, pos_stops, max_walk)

        entry.extend([name1, name2, location1, location2])
        request_data.append(entry)

    # save requests
    request_name = f'requests'
    file_path = os.path.join('data_case', request_name)
    with open(file_path, 'wb') as file:
        pickle.dump(request_data, file)


def json_to_instance(filename, only_comp=False, original_data=False, max_walk=True, gen_req=False):
    """
    Convert json file to DAS instance and make adjustments to it.

    Parameters:
    filename (str): json file
    only_comp (bool): indicates if it is the conventional bus service (CBS) with a fixed tour of fixed stops
    original_data (bool): indicates if we use the original json demand data in the high-demand scenario
    max_walk (bool): is False, if we relax the maximum walking willingness of passengers for the CBS
    gen_req (bool): indicates if we have to generate the request data for the DAS instance,
                    if False we import the pickle file of the demand data

    Return:
    (str): filename of pickle instance after we have saved it
    """
    ##########################################################################################################
    if max_walk:
        max_walking_dist = 333  # meter
    else:
        max_walking_dist = 2800  # meter
    walking_speed = 1.11  # 1.11 m/s = 4 km/h
    driving_speed = 5.56  # 5.56 m/s = 20 km/h
    tw_width = 180  # time window width in seconds
    tw_dist = 1.5  # factor for the segment time
    demand_decrease = 0.4  # share of requests compared to high demand scenario

    # manual adjustment based on visualization
    man_comps = [0, 58, 76, 107, 123, 154]  # set compulsory stops
    man_delete = [1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 19, 39, 75, 102, 88, 105, 97, 136, 143, 144, 145, 146, 147, 148, 153,
                  139, 79, 94, 152, 149, 37, 38, 80, 12, 14, 34, 106, 113, 78, 81, 91]  # delete optional stops
    man_seg = {
        42: 1,
        41: 1,
        40: 1,
        44: 1,
        57: 1,
        87: -1,
        96: 1,
    }  # adjust the segment of these optional stops

    random.seed(1)
    ##########################################################################################################
    data_gen_runtime = time.time()

    data = load_json(filename)

    # convert data
    comps = [window['node_id'] for window in data['time_windows']]

    G = nx.DiGraph()  # whole graph
    G_c = nx.Graph()  # compulsory stops
    if only_comp:
        G_h = [nx.DiGraph() for _ in range(len(comps))]  # graph segments
    else:
        G_h = [nx.DiGraph() for _ in range(len(man_comps))]  # graph segments
    G_s = nx.Graph()  # start nodes of customers
    G_e = nx.Graph()  # end nodes of customers

    seg = 0

    # create first compulsory stop
    name = 'c00'
    node = data['route_nodes'][0]
    location = (node['latitude'], node['longitude'])
    G_c.add_node(name, name2=node['order'], pos=location)
    G_h[seg].add_node(name, name2=node['order'], pos=location)

    # stop infos
    stops = {name: 0}
    locations = {name: location}

    # create stops
    i = 1
    while i < len(data['route_nodes']):
        node = data['route_nodes'][i]
        location = (node['latitude'], node['longitude'])

        # create compulsory stops
        if (only_comp and node['node_id'] in comps) or node['order'] in man_comps:
            name = f'c{seg + 1:02d}'
            G_c.add_node(name, name2=node['order'], pos=location)
            G_h[seg].add_node(name, name2=node['order'], pos=location)
            seg += 1
            G_h[seg].add_node(name, name2=node['order'], pos=location)

        # create optional stops
        else:
            # if not only_comp and node['order'] not in man_delete:
            if node['order'] not in man_delete:
                man_s = man_seg.get(node['order'], 0)
                name = f'o{seg + man_s:02d}_{i:02d}'
                G_h[seg + man_s].add_node(name, name2=node['order'], pos=location)

        stops[name] = i
        locations[name] = location
        i += 1

    # delete last segment (seg after last compulsory stop)
    del G_h[seg]

    # create bus edges
    for i in range(seg):
        for n1 in G_h[i]:
            for n2 in G_h[i]:
                if n1 != f'c{i + 1:02d}' and n2 != f'c{i:02d}' and n1 != n2:
                    # add 10 sek for stopping and letting people get on/off the bus
                    G_h[i].add_edge(n1, n2, weight=round(
                        geodesic(locations[n1], locations[n2]).meters / driving_speed + 10))

        # update graph G
        G = nx.compose(G, G_h[i])

    # bus related graph
    G_b = G.copy()

    # time windows
    a_time_window = [0]
    b_time_window = [0]
    for i in range(len(G_c.nodes()) - 1):
        x = G.get_edge_data(f'c{i:02d}', f'c{i + 1:02d}')['weight']

        # only basic paths is allowed without optional stops
        if only_comp:
            a_time_window.append(a_time_window[-1] + x)
            b_time_window.append(b_time_window[-1] + x)

        # time windows that allow for detours
        else:
            a_time_window.append(a_time_window[-1] + round(x * tw_dist))
            b_time_window.append(a_time_window[-1] + tw_width)

    # big M
    big_M = b_time_window[-1] - b_time_window[0]

    requests = []
    pos_stops_dict = nx.get_node_attributes(G, 'pos')

    # create requests
    if original_data:
        for i in range(len(data['sampled_building_pairs'])):
            # origins
            if i % 2 == 0:
                location1 = (data['sampled_building_pairs'][i]['lat'], data['sampled_building_pairs'][i]['lon'])
                name1 = f's{i // 2:02d}'
                G_s.add_node(name1, pos=location1)
                G.add_node(name1, pos=location1)
                # add edges
                for stop2 in stops_in_walking_distance(location1, pos_stops_dict, max_walking_dist):
                    G.add_edge(name1, stop2,
                               weight=round(geodesic(location1, locations[stop2]).meters / walking_speed))

            # destinations
            else:
                requests.append(f'r{i // 2:02d}')
                location2 = (data['sampled_building_pairs'][i]['lat'], data['sampled_building_pairs'][i]['lon'])
                name2 = f'e{i // 2:02d}'
                G_e.add_node(name2, pos=location2)
                G.add_node(name2, pos=location2)

                # add edges
                for stop1 in stops_in_walking_distance(location2, pos_stops_dict, max_walking_dist):
                    G.add_edge(stop1, name2,
                               weight=round(geodesic(locations[stop1], location2).meters / walking_speed))

                # add penalty edge
                G.add_edge(name1, name2,
                           weight=5000 + round((geodesic(location1, location2).meters / driving_speed)) * tw_dist)

    # adjust data for low demand scenario
    else:
        if gen_req == True:
            generate_demand(data, demand_decrease, pos_stops_dict, max_walking_dist)

        file_path2 = os.path.join('data_case', "requests")
        with open(file_path2, 'rb') as file:
            request_data = pickle.load(file)

        for req, name1, name2, location1, location2 in request_data:
            requests.append(req)

            # add nodes
            G_s.add_node(name1, pos=location1)
            G.add_node(name1, pos=location1)
            G_e.add_node(name2, pos=location2)
            G.add_node(name2, pos=location2)

            # add edges
            for stop2 in stops_in_walking_distance(location1, pos_stops_dict, max_walking_dist):
                G.add_edge(name1, stop2,
                           weight=round(geodesic(location1, locations[stop2]).meters / walking_speed))

            # add edges
            for stop1 in stops_in_walking_distance(location2, pos_stops_dict, max_walking_dist):
                G.add_edge(stop1, name2,
                           weight=round(geodesic(locations[stop1], location2).meters / walking_speed))

            # add penalty edge
            if only_comp:
                # use walking time as penalty since the tour does not get changed either way
                G.add_edge(name1, name2,
                           weight=round((geodesic(location1, location2).meters / walking_speed)))
            else:
                # add bigger penalty for DAS
                G.add_edge(name1, name2,
                           weight=5000 + round((geodesic(location1, location2).meters / driving_speed)) * tw_dist)

    # demand
    demand = {x: 1 for x in requests}

    instance = gurobi_model.Instance(G, G_b, G_c, G_s, G_e, G_h, demand, requests, a_time_window, b_time_window, big_M)

    data_gen_runtime = time.time() - data_gen_runtime
    print(f'It took {data_gen_runtime} seconds to create the data')

    # save instance
    instance_name = f'munich_{filename[5:-5]}'
    if only_comp:
        instance_name = f'fix_{instance_name}'
    file_path = os.path.join('data_case', instance_name)
    with open(file_path, 'wb') as file:
        pickle.dump(instance, file)

    return instance_name


def optimize(filename):
    """
    Solve this instance with P100 and save result information.

    Parameters:
    filename (str): instance pickle file

    Return:
    (str): filename of pickle result data after we have saved it
    """
    # load instance
    file_path = os.path.join('data_case', filename)
    with open(file_path, 'rb') as file:
        instance = pickle.load(file)

    heuristic_runtime = time.time()
    heuristic_name = 'grasp for path'
    # find route via heuristic
    result_data, _ = heuristic_path.grasp_for_path_generation(instance, [1, 5, 100], case_study=True)
    heuristic_runtime = time.time() - heuristic_runtime
    print(f'It took {heuristic_runtime} seconds to perform the heuristic {heuristic_name}')

    # save result data
    result_name = f'result_{filename[7:]}'
    if filename[0] == 'f':
        result_name = f'fix_result_{filename[11:]}'

    file_path = os.path.join('data_case', result_name)
    with open(file_path, 'wb') as file:
        pickle.dump(result_data, file)

    return result_name


def visualize(filename, only_comp=False, no_req=False, result=None):
    """
    Visualize a map.

    Parameters:
    filename (str): instance pickle file
    only_comp (bool): indicates if it is the conventional bus service (CBS) with a fixed tour of fixed stops
    no_req (bool): indicates if we leave the requests out of the visualization
    result (str): filename of pickle result data

    Saves the map and opens it 
    """
    # load instance
    file_path = os.path.join('data_case', filename)
    with open(file_path, 'rb') as file:
        instance = pickle.load(file)

    if result:
        file_path = os.path.join('data_case', result)
        with open(file_path, 'rb') as file:
            result_data = pickle.load(file)

    map = folium.Map(location=instance.G.nodes[instance.C[0]]['pos'], zoom_start=14)

    if only_comp:
        colors = ['black'] * 44
    else:
        colors = ['black', 'dimgrey', 'black', 'dimgray', 'black', 'grey', 'black', 'grey', ]

    # compulsory stops
    i = 0
    comp_points = []
    for comp in instance.C:
        icon_square = BeautifyIcon(
            icon_shape='rectangle-dot',
            border_color='black',
            border_width=8,
            background_color='transparent'
        )
        loc = instance.G.nodes[comp]['pos']
        name2 = instance.G.nodes[comp]['name2']
        comp_points.append(loc)
        folium.Marker(location=loc, tooltip=f'{comp}-{name2}', icon=icon_square, z_index_offset=1000).add_to(map)
        i += 1

        if only_comp:
            folium.PolyLine(comp_points, color="black", weight=2.5, opacity=1).add_to(map)

    if not only_comp:
        # optional stops grouped in segments
        c = 0
        for G in instance.G_h:
            seg_opt = []
            for node, attrs in G.nodes(data=True):
                icon_circle = BeautifyIcon(
                    icon_shape='circle-dot',
                    border_color=colors[c],
                    border_width=6,
                    background_color='transparent'
                )
                if node not in instance.C:
                    folium.Marker(location=attrs['pos'], tooltip=f'{node}-{attrs["name2"]}', icon=icon_circle).add_to(
                        map)
                    seg_opt.append(attrs['pos'])

            c += 1
            print(c)

    # add markers for requests
    if not no_req:
        for n in instance.N_S:
            icon_start = BeautifyIcon(
                icon='home',
                inner_icon_style='color:royalblue;font-size:14px;',
                background_color='transparent',
                border_color='transparent',
            )
            folium.Marker(location=instance.G.nodes[n]['pos'], tooltip=n, icon=icon_start).add_to(map)

        for n in instance.N_E:
            icon_end = BeautifyIcon(
                icon='flag',
                inner_icon_style='color:mediumblue;font-size:14px;',
                background_color='transparent',
                border_color='transparent',
            )
            folium.Marker(location=instance.G.nodes[n]['pos'], tooltip=n, icon=icon_end, z_index_offset=1).add_to(map)

    # print tour
    if result:
        for n1, n2 in result_data.x_values:
            folium.PolyLine(locations=[instance.G.nodes[n1]['pos'], instance.G.nodes[n2]['pos']],
                            color='black', weight=2.5, opacity=1).add_to(map)

        # some information on the result
        print(result_data.path_duration)
        print(result_data.departure_time)
        print(result_data.waiting_bus)
        print(f'min tour length: {instance.a[-1]}, max tour length: {instance.b[-1]}')
        print(instance.a)
        print(instance.b)
        print(f'onboard: {result_data.user_onboard}')
        print(f'penalty: {result_data.user_penalty}')
        print(f'Request accepted: {len(result_data.user_onboard)}, Request rejected: {len(result_data.user_penalty)}')
        print(f'Stops: {len(result_data.x_values) + 1}')

    # save and open map
    map.save('map.html')
    webbrowser.open('map.html')


if __name__ == "__main__":
    # Load the JSON file into a dictionary
    file = '55_15_58_00_1_200_200_0.json'
    filename = 'data\\' + file

    # DAS
    # instance_filename = json_to_instance(filename, gen_req=False)
    instance_filename = f'munich_{file[:-5]}'
    result_name = optimize(instance_filename)
    # result_name = f'result_{file[:-5]}'
    visualize(instance_filename, result=result_name)

    # CBS
    # instance_filename = json_to_instance(filename, only_comp=True, max_walk=False)
    instance_filename = f'fix_munich_{file[:-5]}'
    result_name = optimize(instance_filename)
    # result_name = f'fix_result_{file[:-5]}'
    visualize(instance_filename, result=result_name, only_comp=True)
