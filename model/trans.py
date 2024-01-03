import csv
from geopy.distance import geodesic


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

    def print_stops(self, line_id, from_stop, to_stop):
        line = self.lines[line_id]
        cat = line.category
        total_time = 0  # 公共交通总时间 min
        speed = 0  # 交通工具的速度 m/min  1km/h=16.7m/min
        if cat == '301' or cat == '304':  # 公交
            speed = 30 * 16.7
        elif cat == '302':  # BRT
            speed = 40 * 16.7
        elif cat == '303':  # 轮船，默认一趟15min
            total_time = 15
        elif cat == '305':  # 地铁
            speed = 50 * 16.7
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
        for i in range(abs(start_index-end_index+1)):
            cur_station = line.stations[temp_list[start_index+i*sign][1]]
            if cur_station.station_id == to_stop:
                print(cur_station.station_name)
            else:
                print(cur_station.station_name, " -> ", end="")
            if cur_station.station_id == last_station.station_id:
                continue
            else:
                distance = geodesic((last_station.latitude, last_station.longitude), (cur_station.latitude, cur_station.longitude)).m
                if speed != 0:
                    total_time += distance / speed
                # print(distance)
            last_station = cur_station
        print("Total time: ", total_time, "min")
        return total_time

    def get_line_name(self, line_id):
        if line_id in self.lines:
            return self.lines[line_id].line_name
        else:
            return None


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
    # ========== 使用Dijikstrea算法计算路径
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
    total_time = 0
    for each_route in solution:
        print("在 " + station_manager.get_station_name(each_route.from_stop) + " 乘坐 " + str(
            line_manager.get_line_name(each_route.line_number)) + "号线 到 " + station_manager.get_station_name(each_route.to_stop) + "(" + str(each_route.stops) + "站)")
        total_time += line_manager.print_stops(each_route.line_number, each_route.from_stop, each_route.to_stop)
        total_time += 1     # 换乘增加一分钟
    print('Total time: ', total_time, 'min')


def get_near_station(station_manager, lat1, long1, lat2, long2):
    start = ''
    terminal = ''
    min_start = 9999
    min_terminal = 9999
    for key in station_manager.stations:
        each = station_manager.stations[key]
        dis_start = geodesic((float(lat1), float(long1)), (float(each.latitude), float(each.longitude))).m
        if min_start > dis_start:
            min_start = dis_start
            start = each.station_id
        dis_terminal = geodesic((float(lat2), float(long2)), (float(each.latitude), float(each.longitude))).m
        if min_terminal > dis_terminal:
            min_terminal = dis_terminal
            terminal = each.station_id
    return start, terminal


if __name__ == '__main__':
    # 初始化站点和线路
    station_manager, line_manager = initiate_manager('../data/transportation.csv')
    # 找到离起点和终点最近的公交站
    # start = '35288'     # 海沧湾公园[地铁]
    # terminal = '34782'  # 湖滨东路[地铁]
    start, terminal = get_near_station(station_manager, '24.454744', '118.116097', '24.43582', '118.116178') # 园博园到环岛路
    # 获取公共交通路线（含换乘）
    get_bus_route(station_manager, line_manager, start, terminal)
