import csv

import numpy as np
from geopy.distance import geodesic
import math
from data_process import *


class Line:
    def __init__(self, line_id, line_name, category):
        self.line_id = line_id
        self.line_name = line_name
        self.category = category
        self.station_ids = []
        self.stations = {}

    def add_station(self, station):
        self.station_ids.append(station.station_id)
        self.stations[station.station_id] = station


class LineManager:
    def __init__(self):
        self.lines = {}

    def add_line(self, line_id, line_name, category, station):
        if line_id in self.lines:
            self.lines[line_id].add_station(station)
        else:
            line = Line(line_id, line_name, category)
            line.add_station(station)
            self.lines[line_id] = line

    def get_best_route(self, from_station, to_station, lines):
        route = Route()
        route.from_stop = from_station
        route.to_stop = to_station
        route.stops = 9999
        if len(lines) == 0:
            route.stops = 9999
            return route
        else:
            for each_line in lines:
                line = self.lines[each_line]
                start_index = 0
                stop_index = 0
                for i in range(0, len(line.station_ids)):
                    if line.station_ids[i] == from_station:
                        start_index = i
                    elif line.station_ids[i] == to_station:
                        stop_index = i
                stops = abs(start_index - stop_index)
                if stops < route.stops:
                    route.stops = stops
                    route.line_number = line.line_id
        return route

    def get_stops(self, line_id, from_stop, to_stop):
        line = self.lines[line_id]
        cat = line.category
        total_time = 0  # 公共交通总时间 min
        speed = 0  # 交通工具的速度 m/min  1km/h=16.7m/min
        fee = 0
        if cat == '301' or cat == '304':  # 公交
            speed = 30 * 16.7
            fee = 2
        elif cat == '302':  # BRT
            speed = 40 * 16.7
            fee = 3
        elif cat == '303':  # 轮船，默认一趟15min
            total_time = 15
            fee = 35
        elif cat == '305':  # 地铁
            speed = 50 * 16.7
            fee = 4
        start_index = 0
        end_index = 0
        for i in range(0, len(line.station_ids) - 1):
            if line.station_ids[i] == from_stop:
                start_index = i
            elif line.station_ids[i] == to_stop:
                end_index = i
        sign = 1
        if start_index > end_index:
            sign = -1
        last_station = line.stations[from_stop]
        temp_list = list(enumerate(line.stations))
        line_route = [line.line_name]
        for i in range(abs(start_index-end_index+1)):
            cur_station = line.stations[temp_list[start_index+i*sign][1]]
            line_route.append(cur_station.station_name)
            if cur_station.station_id == last_station.station_id:
                continue
            else:
                distance = geodesic((last_station.latitude, last_station.longitude), (cur_station.latitude, cur_station.longitude)).m
                if speed != 0:
                    total_time += distance / speed
                # print(distance)
            last_station = cur_station
        return line_route, total_time, fee

    def get_line_name(self, line_id):
        if line_id in self.lines:
            return self.lines[line_id].line_name
        else:
            return None

    def print_all_dis(self):
        for key in self.lines:
            each = self.lines[key]
            print(each.line_name)
            last = each.stations[each.station_ids[0]]
            for key2 in each.stations:
                cur = each.stations[key2]
                if last == cur:
                    continue
                else:
                    distance = geodesic((last.latitude, last.longitude), (cur.latitude, cur.longitude)).km
                    with open('../data/test3.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([last.station_id, last.station_name, cur.station_id, cur.station_name, distance])
                last = cur


class Station:
    def __init__(self, station_id, station_name, longitude, latitude):
        self.station_id = station_id
        self.station_name = station_name
        self.longitude = longitude
        self.latitude = latitude
        self.lines = []

    def add_line(self, line_id):
        self.lines.append(line_id)


class StationManager:
    def __init__(self):
        self.stations = {}
        self.station_ids = set()

    def add_station(self, station_id, station_name, longitude, latitude, line_id):
        self.station_ids.add(station_id)
        if station_id in self.stations:
            self.stations[station_id].add_line(line_id)
        else:
            station = Station(station_id, station_name, longitude, latitude)
            station.add_line(line_id)
            self.stations[station_id] = station

    def get_same_lines(self, from_station, to_station):
        line_numbers = []
        for each_line in from_station.lines:
            if each_line in to_station.lines:
                line_numbers.append(each_line)
        return line_numbers

    def get_station_name(self, station_id):
        if station_id in self.station_ids:
            return self.stations[station_id].station_name
        else:
            return None


class Route:
    from_stop = ""
    to_stop = ""
    line_number = 0
    stops = 9999


def initiate_manager(filename):
    station_manager = StationManager()
    line_manager = LineManager()
    with open(filename) as file:
        csv_reader = csv.reader(file)
        next(csv_reader)    # 跳过表头
        for row in csv_reader:
            line_id = row[0]
            line_name = row[1]
            longitude = row[2]
            latitude = row[3]
            # seq = row[4] 不需要了，因为csv默认按照站点顺序排的
            station_id = row[5]
            station_name = row[6]
            category = row[7]
            station_manager.add_station(station_id, station_name, longitude, latitude, line_id)
            line_manager.add_line(line_id, line_name, category, Station(station_id, station_name, longitude, latitude))
    return station_manager, line_manager


def get_bus_route(station_manager, line_manager, start, terminal):
    stations = []
    station_index = {}
    v_matrix = []   # 邻接矩阵
    book = []
    dis = []
    n = 0  # 顶点数
    # ========== 初始化邻接矩阵
    index = 0
    for each in station_manager.stations.keys():
        station_index[each] = index
        index += 1
        stations.append(station_manager.stations[each])
    for i in range(0, len(stations)):
        v_matrix.append([])
        for j in range(0, len(stations) - 1):
            same_lines = station_manager.get_same_lines(stations[i], stations[j])
            v_matrix[i].append(line_manager.get_best_route(stations[i].station_id, stations[j].station_id, same_lines))
    start_index = station_index[start]
    terminal_index = station_index[terminal]
    # ========== 使用Dijkstra算法计算路径
    # 初始化dis数组
    for i in range(0, len(v_matrix) - 1):
        dis.append((v_matrix[start_index][i].stops, [v_matrix[start_index][i]]))
    for i in range(0, len(v_matrix) - 1):
        book.append(0)
    book[0] = 1
    n = len(stations)
    u = 0
    for i in range(0, n - 1):
        min = (9999, [Route()])
        for j in range(0, n - 1):
            if book[j] == 0 and dis[j][0] < min[0]:
                min = (dis[j][0], dis[j][1])
                u = j
        book[u] = 1
        bias = 30
        for v in range(0, n - 1):
            if v_matrix[u][v].stops <= 9999:
                if book[v] == 0 and dis[v][0] > (dis[u][0] + v_matrix[u][v].stops + bias):
                    a = []
                    for each in dis[u][1]:
                        a.append(each)
                    a.append(v_matrix[u][v])
                    dis[v] = (dis[u][0] + v_matrix[u][v].stops, a)
    solution = dis[terminal_index][1]
    total_route = {}    # 以线路名称为key，途径的站点list为value的字典
    total_time = 0
    total_fee = 0
    for each_route in solution:
        # print("在 " + station_manager.get_station_name(each_route.from_stop) + " 乘坐 " + str(line_manager.get_line_name(each_route.line_number)) + "号线 到 " + station_manager.get_station_name(each_route.to_stop) + "(" + str(each_route.stops) + "站)")
        temp_route, temp_time, temp_fee = line_manager.get_stops(each_route.line_number, each_route.from_stop, each_route.to_stop)
        total_route[temp_route[0]] = temp_route[1:]
        total_time += temp_time
        total_time += 1     # 换乘增加一分钟
        total_fee += temp_fee
    return total_route, total_time, total_fee


def get_other_route(lat1, long1, lat2, long2):
    distance = geodesic((float(lat1), float(long1)), (float(lat2), float(long2))).m
    distance *= 2
    taxi_time = distance / 360
    taxi_fee = 10     # 10r起步，超过3公里每公里2r
    if distance > 3000:
        taxi_fee += 2 * math.ceil((distance - 3000) / 1000)
    walk_time = distance / 80
    return taxi_time, taxi_fee, walk_time


def get_near_station(station_manager, lat1, long1):
    near = ''
    min_dis = 9999
    for key in station_manager.stations:
        each = station_manager.stations[key]
        cur_dis = geodesic((float(lat1), float(long1)), (float(each.latitude), float(each.longitude))).m
        if min_dis > cur_dis:
            min_dis = cur_dis
            near = each.station_id
    return near


def cal_attraction():
    G, user_visit, spot_visit = build_traj_graph()
    user_value = {}     # 用户的经验值=访问过的景点的兴趣值之和
    last_user_value = {}
    spot_value = {}     # 景点的兴趣值=被访问过的用户经验值之和
    last_spot_value = {}
    # 随机初始化
    for key in user_visit:
        last_user_value[key] = user_value[key] = 1
    for key in spot_visit:
        last_spot_value[key] = spot_value[key] = 1
    while True:
        for key in spot_visit:
            users = spot_visit[key]
            spot_value[key] = 0
            for each in users:
                spot_value[key] += last_user_value[each]
        # 除以最大值将其标准化
        spot_value_max = max(list(spot_value.values()))
        for key in spot_value:
            spot_value[key] /= spot_value_max
        for key in user_visit:
            spots = user_visit[key]
            user_value[key] = 0
            for each in spots:
                user_value[key] += last_spot_value[each]
        user_value_max = max(list(user_value.values()))
        for key in user_value:
            user_value[key] /= user_value_max
        # 结果收敛时结束算法
        user_diff = []
        spot_diff = []
        for key in user_visit:
            user_diff.append(abs(user_value[key]-last_user_value[key]))
        for key in spot_visit:
            spot_diff.append(abs(spot_value[key]-last_spot_value[key]))
        diff = sum(user_diff) + sum(spot_diff)
        if diff < 0.01:
            break
        else:
            for key in user_visit:
                last_user_value[key] = user_value[key]
            for key in spot_visit:
                last_spot_value[key] = spot_value[key]
    spot_id = list(spot_value.keys())
    attraction = np.zeros((len(spot_id), len(spot_id)))
    adj = np.array(nx.adjacency_matrix(G, nodelist=G.nodes()).todense())
    row_sum = np.sum(adj, axis=1)
    col_sum = np.sum(adj, axis=0)
    for i in range(len(spot_id)):
        for j in range(len(spot_id)):
            if i == j:
                continue
            if row_sum[i] == 0 or col_sum[j] == 0:
                attraction[i][j] = 0
                continue
            user_sum = 0
            users = spot_visit[spot_id[i]]
            for each in users:
                if [spot_id[i], spot_id[j]] in user_visit[each]:
                    user_sum += user_value[each]
            # A→B的吸引力=A→B的次数/A的总出度*A的兴趣值+A→B的次数/B的总入度*B的兴趣值+访问过该顺序的用户经验值之和
            attraction[i][j] = adj[i][j] / row_sum[i] * spot_value[spot_id[i]] + adj[i][j] / col_sum[j] * spot_value[spot_id[j]] + user_sum
    np.savetxt('../data/attraction.csv', attraction, delimiter=',')
    return attraction


def cal_heuristics():
    spots = load_poi_features('../data/spot.csv')
    heuristics = np.zeros((spots.shape[0], spots.shape[0]))
    near = np.loadtxt('../data/near_station.csv', delimiter=',')
    near_stations = list(near[:, 0])
    for i in range(len(near_stations)):
        near_stations[i] = str(int(near_stations[i]))
    near_time = list(near[:, 1])
    G = build_trans_graph()
    np.savetxt(os.path.join('../data/trans_graph_A.csv'), nx.adjacency_matrix(G, nodelist=G.nodes()).todense(), delimiter=',')
    for i in range(spots.shape[0]):
        for j in range(i+1, spots.shape[0]):
            heuristics[i][j] += near_time[i]
            heuristics[i][j] += near_time[j]
            try:
                heuristics[i][j] += nx.dijkstra_path_length(G, int(near_stations[i]), int(near_stations[j])) / 500
            except:     # 找不到路径时
                heuristics[i][j] += geodesic((spots[i][6], spots[i][7]), (spots[j][6], spots[j][7])).m / 40
            heuristics[j][i] = heuristics[i][j]
    np.savetxt('../data/heuristics.csv', heuristics, delimiter=',')
    return heuristics


# Improved A Star Algorithm using TPN
def AStarPlus():
    # ========== 构建可预测成本g
    distance = load_distance('../data/distance.csv')
    attraction = load_matrix('../data/attraction.csv')
    # 归一化处理
    dis_min = distance.min()
    k1 = 10 / (distance.max() - dis_min)
    dis_new = np.where(True, k1 * (distance - dis_min), distance)
    attract_min = attraction.min()
    k2 = 10 / (attraction.max() - attract_min)
    attract_new = np.where(attraction > 0, 10 - k2 * (attraction - attract_min), attraction)    # 吸引力越小，代价越大
    g = 0.6 * dis_new + 0.4 * attract_new
    # ========== 构建估计成本h
    route = load_matrix('../data/heuristics.csv')
    route_min = route.min()
    k3 = 10 / (route.max() - route_min)
    h = np.where(route > 0, k3 * (route - route_min), route)


if __name__ == '__main__':
    '''
    # 初始化站点和线路
    station_manager, line_manager = initiate_manager('../data/transportation.csv')
    # line_manager.print_all_dis()
    # 找到离起点和终点最近的公交站
    # start = '35288'     # 海沧湾公园[地铁]
    # terminal = '34782'  # 湖滨东路[地铁]
    s_lat = '24.454744'
    s_long = '118.116097'
    t_lat = '24.43582'
    t_long = '118.116178'
    start = get_near_station(station_manager, s_lat, s_long) # 园博园到环岛路
    terminal = get_near_station(station_manager, t_lat, t_long)
    # 获取公共交通路线（含换乘）
    bus_route, bus_time, bus_fee = get_bus_route(station_manager, line_manager, start, terminal)
    print('---------- Bus Route ----------')
    for key in bus_route:
        print('From ', bus_route[key][0], ' take ', key, ' to ', bus_route[key][len(bus_route[key]) - 1])
        for i in range(len(bus_route[key]) - 1):
            print(bus_route[key][i], ' > ', end='')
        print(bus_route[key][len(bus_route[key]) - 1])
    print('Estimated total time: ', bus_time, 'min')
    print('Estimated total fee: ', bus_fee, 'rmb')
    # 获取打的和步行的路线
    taxi_time, taxi_fee, walk_time = get_other_route(s_lat, s_long, t_lat, t_long)
    print('---------- Taxi Route ----------')
    print('Estimated total time: ', taxi_time, 'min')
    print('Estimated total fee: ', taxi_fee, 'rmb')
    print('---------- Walk Route ----------')
    print('Estimated total time: ', walk_time, 'min')
    '''
    # cal_attraction()
    AStarPlus()