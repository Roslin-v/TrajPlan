import csv
import json
import numpy as np
import requests
from geopy.distance import geodesic
import math
from data_process import *


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
    with open('../data/interest.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in spot_value.items():
            writer.writerow(row)
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


def get_cost():
    distance = load_matrix('../data/heuristics.csv')
    attraction = load_matrix('../data/attraction.csv')
    # 归一化处理
    dis_min = distance.min()
    k1 = 20 / (distance.max() - dis_min)
    dis_new = np.where(distance > 0, k1 * (distance - dis_min), distance)
    attract_min = attraction.min()
    k2 = 20 / (attraction.max() - attract_min)
    attract_new = np.where(attraction > 0, 20 - k2 * (attraction - attract_min), 100)  # 吸引力越小，代价越大
    cost = 0.6 * dis_new + 0.4 * attract_new
    for i in range(cost.shape[0]):  # 对角线（自己）的转移概率无限大
        cost[i][i] = 99
    return cost


class PlanManager:
    def __init__(self, constraint):
        self.constraint = constraint    # 约束
        self.all_cost = get_cost()      # 各个景点之间的转移代价
        self.plan = {}                  # 计划 {day1: [poi_id, poi_name, ...]}
        self.plan_situ = {}             # 计划情况 {day1: [是否需要补充行程, [已有poi]]}
        self.trans = {'bus': [], 'taxi': [], 'walk': []}
        self.score = 0
        self.spot_feat = load_poi_features('../data/spot.csv')
        self.food_feat = load_poi_features('../data/food.csv')

    def reinitial(self, constraint):
        # 清空原来的计划
        self.constraint = constraint
        self.plan = {}
        self.plan_situ = {}
        self.trans = {'bus': [], 'taxi': [], 'walk': []}
        self.score = 0

    def knapsack(self, budgets, times, B, T):
        # ========== 获取背包的各项初始化数据
        nodes = self.constraint['select-spot']
        importance = []
        raw_int = load_matrix('../data/interest.csv')
        interest = []
        comment = []
        for each in nodes:
            interest.append(raw_int[each - 10001][1])
            comment.append(self.spot_feat[each - 10001][4])
        interest_min = min(interest)
        comment_min = min(comment)
        k1 = 30 / (max(interest) - interest_min)
        k2 = 70 / (max(comment) - comment_min)
        # 重要程度 = 40%景点兴趣度 + 60%评论人数
        for i in range(len(interest)):
            interest[i] = k1 * (interest[i] - interest_min)
            comment[i] = k2 * (comment[i] - comment_min)
            importance.append(interest[i] + comment[i])
        # ========== 动态规划求解双限制（时间、预算）的0-1背包问题
        n = len(importance)
        dp = [[[0 for _ in range(T + 1)] for _ in range(B + 1)] for _ in range(n + 1)]
        item_selection = [[[False for _ in range(T + 1)] for _ in range(B + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for b in range(1, B + 1):
                for t in range(1, T + 1):
                    if budgets[i - 1] > b or times[i - 1] > t:
                        dp[i][b][t] = dp[i - 1][b][t]
                    else:
                        not_taking_item = dp[i - 1][b][t]
                        taking_item = dp[i - 1][int(b - budgets[i - 1])][int(t - times[i - 1])] + importance[i - 1]
                        if taking_item > not_taking_item:
                            dp[i][b][t] = taking_item
                            item_selection[i][b][t] = True
                        else:
                            dp[i][b][t] = not_taking_item
        # ========== 回溯找到所选择的景点
        selected_items = []
        b, t = B, T
        for i in range(n, 0, -1):
            if item_selection[i][b][t]:
                selected_items.append(i - 1)
                b -= budgets[i - 1]
                t -= times[i - 1]
        selected_items.reverse()  # 逆序以得到原始顺序
        return selected_items

    def check_constraint(self):
        nodes = self.constraint['select-spot']
        budgets = []
        times = []
        for each in nodes:
            budgets.append(int(self.spot_feat[each - 10001][5]))
            if self.spot_feat[each - 10001][10] in nodes:
                times.append(0)
            else:
                times.append(int(self.spot_feat[each - 10001][8] + 1))   # 多加1小时，粗略算作交通和吃饭时间
        B = int(self.constraint['user-budget'] / 2)  # 留点预算给餐厅
        T = int(self.constraint['user-time'])
        if T >= 24:  # 如果用户选择超过一天，一天之中最多12h可以实际游玩
            T /= 2
            T = int(T)
        if sum(budgets) > B / 2 or sum(times) > T:
            new_pois_index = self.knapsack(budgets, times, B, T)
            new_pois = []
            for each in new_pois_index:
                new_pois.append(nodes[each])
            self.constraint['select-spot'] = new_pois

    def ant_colony(self):
        nodes = self.constraint['select-spot']
        nodes.append(10000)      # 增加一个虚拟点
        node_count = len(nodes)  # 节点数量
        # POI ID转为索引
        for i in range(node_count):
            nodes[i] %= 10000
            nodes[i] -= 1
        he = load_matrix('../data/heuristics.csv')
        trans_time = np.zeros((node_count-1, node_count-1))
        # 构建转移代价矩阵
        cost = np.zeros((node_count, node_count))
        for i in range(node_count - 1):
            for j in range(node_count - 1):
                cost[i][j] = self.all_cost[nodes[i]][nodes[j]]
                trans_time[i][j] = round(he[nodes[i]][nodes[j]] / 60 * 2) / 2
        # 虚拟点和其他点的转移代价都为0，但是由于后续要用到cost的倒数，将其设置为一个很小的值
        for i in range(node_count):
            cost[node_count-1][i] = 0.001
            cost[i][node_count-1] = 0.001
        # ========== 设置参数
        ant_count = 50              # 蚂蚁数量
        alpha = 1                   # 信息素重要程度因子
        beta = 2                    # 启发函数重要程度因子
        rho = 0.1                   # 挥发速度
        iter = 0                    # 迭代初始值
        max_iter = 200              # 最大迭代值
        Q = 1
        pheromone = np.ones((node_count, node_count))   # 初始信息素矩阵，全是为1组成的矩阵
        candidate = np.zeros((ant_count, node_count)).astype(int)   # 候选集列表，存放蚂蚁的路径
        path_best = np.zeros((max_iter, node_count))    # 每次迭代后的最优路径
        cost_best = np.zeros(max_iter)              # 存放每次迭代的最优距离
        etable = 1.0 / cost         # 倒数矩阵
        # ======== 开始迭代
        while iter < max_iter:
            # Step1: 蚂蚁初始点选择（虚拟点）
            candidate[:, 0] = node_count - 1
            length = np.zeros(ant_count)  # 每次迭代的N个蚂蚁的距离值
            # Step2: 选择下一个节点
            for i in range(ant_count):
                # 移除已经访问的第一个元素
                un_visit = list(range(node_count))  # 列表形式存储没有访问的节点编号
                visit = candidate[i, 0]             # 当前所在点,第i个蚂蚁在第一个节点
                un_visit.remove(visit)              # 在未访问的节点中移除当前开始的点
                for j in range(1, node_count):      # 访问剩下的节点
                    trans_prob = np.zeros(len(un_visit))  # 每次循环都更改当前没有访问的节点的转移概率矩阵
                    # 下一节点的概率函数
                    for k in range(len(un_visit)):
                        # 计算当前节点到剩余节点的（信息素浓度^alpha）*（节点适应度的倒数）^beta
                        trans_prob[k] = np.power(pheromone[visit][un_visit[k]], alpha) * np.power(etable[visit][un_visit[k]], beta)
                    # 累计概率，轮盘赌选择
                    total_prob = (trans_prob / sum(trans_prob)).cumsum()
                    total_prob -= np.random.rand()
                    # 求出离随机数产生最近的索引值
                    k = un_visit[list(total_prob > 0).index(True)]
                    # 下一个访问节点的索引值
                    candidate[i, j] = k
                    un_visit.remove(k)
                    length[i] += cost[visit][k]
                    visit = k  # 更改出发点，继续选择下一个到达点
                length[i] += cost[visit][candidate[i, 0]]  # 最后一个节点和第一个节点的距离值也要加进去
            # Step3: 更新路径
            # 如果迭代次数为一次，那么无条件让初始值代替path_best,cost_best.
            if iter == 0:
                cost_best[iter] = length.min()
                path_best[iter] = candidate[length.argmin()].copy()
            else:
                # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
                if length.min() > cost_best[iter - 1]:
                    cost_best[iter] = cost_best[iter - 1]
                    path_best[iter] = path_best[iter - 1].copy()
                else:  # 当前解比之前的要好，替换当前解和路径
                    cost_best[iter] = length.min()
                    path_best[iter] = candidate[length.argmin()].copy()
            # Step 4: 更新信息素
            pheromone_change = np.zeros((node_count, node_count))
            for i in range(ant_count):
                for j in range(node_count - 1):
                    # 当前路径之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
                    pheromone_change[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
                # 最后一个节点和第一个节点的信息素增加量
                pheromone_change[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
            pheromone = (1 - rho) * pheromone + pheromone_change
            iter += 1
        route = []
        for i in range(1, len(path_best[-1])):
            route.append(int(path_best[-1][i]))
        spots = []
        for i in range(node_count - 1):
            spots.append(self.spot_feat[nodes[i]])
        belong = {}
        for i in range(len(route)):
            try:
                index = route.index(nodes.index(spots[route[i]][10] % 10000 - 1))
            except:
                continue
            if index in belong:
                belong[index].append(i)
            else:
                belong[index] = [i]
        # 把最大的地区换到最前面
        bk = []
        for key in belong:
            bk.append(key)
        for each in bk:
            b = min(belong[each])
            if each > b:
                route[each], route[b] = route[b], route[each]
                belong[b] = belong[each]
                i = belong[b].index(b)
                belong[b][i] = each
                del belong[each]
        cur_day = 1
        cur_time = 9
        plan = {}   # {day1: id, name, start_time, end_time, price}
        i = 0
        # ========== 切分天数
        while i != len(route):
            end_time = cur_time + spots[route[i]][8]
            if (end_time > 17 and spots[route[i]][9] == 0) or end_time > 21:
                cur_day += 1
                cur_time = 9
                end_time = cur_time + spots[route[i]][8]
            if cur_day not in plan:
                plan[cur_day] = []
            plan[cur_day].append([spots[route[i]][0], spots[route[i]][1], cur_time, end_time, spots[route[i]][5]])
            if i in belong:
                for j in range(len(belong[i])):
                    temp_time = cur_time + spots[route[belong[i][j]]][8]
                    plan[cur_day].append([spots[route[belong[i][j]]][0], spots[route[belong[i][j]]][1], cur_time, temp_time, spots[route[belong[i][j]]][5]])
                    cur_time = temp_time
                if j != (len(belong[i]) - 1):
                    cur_time += trans_time[belong[i][j]][belong[i][j+1]]
                i = max(belong[i])
            cur_time = end_time
            if i != (len(route) - 1):
                cur_time += trans_time[route[i]][route[i+1]]
            i += 1
        self.plan = plan
        for key in plan:
            p = plan[key]
            self.plan_situ[key] = [0, []]  # {day1: 是否需要补充行程, [已有poi]}
            for each in p:
                self.plan_situ[key][1].append(each[0])
            if p[-1][-2] < 18:  # 结束时间小于18点
                self.plan_situ[key][0] = 1
        plan_time = (len(plan) - 1) * 24
        if int(list(plan.items())[-1][1][-1][0] / 10000) == 1:
            plan_time += list(plan.items())[-1][1][-1][-2]
        else:
            plan_time += list(plan.items())[-1][1][-2][-2]
        # Todo: 如果超时（救命这个写得好奇怪啊呜呜呜）
        if plan_time > self.constraint['user-time']:
            # 先看能不能拼接已有的行程
            key1 = key2 = 0
            keys = []
            for key in self.plan_situ.keys():
                keys.append(key)
            for key in keys:
                if self.plan_situ[key][0] == 1:
                    if key1 == 0:
                        key1 = key
                    else:
                        key2 = key
                        if self.plan[key1][-1][-2] + self.plan[key2][-1][-2] - self.plan[key2][0][2] < 21:
                            for each in self.plan[key2]:
                                each[2] += (self.plan[key1][-1][-2] - self.plan[key1][0][2] + 1)
                                each[3] += (self.plan[key1][-1][-2] - self.plan[key1][0][2] + 1)
                            self.plan[key1] += self.plan[key2]
                            self.plan_situ[key1][0] = 0
                            del self.plan[key2]
                            del self.plan_situ[key2]
            # 重新计算时间
            plan_time = (len(self.plan) - 1) * 24 + list(self.plan.items())[-1][1][-1][-2]
        self.constraint['all-time'] = plan_time
        plan_fee = 0
        for key in self.plan:
            for each in self.plan[key]:
                plan_fee += each[4]
        self.constraint['all-budget'] = plan_fee

    def print_plan(self):
        for key in self.plan:
            print('>>> Day', key)
            p = self.plan[key]
            for each in p:
                if int(each[0] / 10000) == 1:    # 景点
                    print(each[1], ':', end=' ')
                    print(math.floor(each[2]), end='')
                    if math.floor(each[2]) != each[2]:
                        print(':30-', end='')
                    else:
                        print(':00-', end='')
                    print(math.floor(each[3]), end='')
                    if math.floor(each[3]) != each[3]:
                        print(':30', end=' ')
                    else:
                        print(':00', end=' ')
                    if each[4] != 0:
                        print('门票:', each[4], '元')
                    else:
                        print('')
                else:   # 餐厅
                    print(each[1], ': 评分', each[2], '(', end='')
                    print(each[3], end='')
                    print(')', end=' ')
                    if each[4] != 0:
                        print('人均:', each[4], '元')
                    else:
                        print('')
        print('Total time:\t', self.constraint['all-time'], '小时')
        print('Total fee:\t', self.constraint['all-budget'], '元')

    def recommend_food(self, point1, point2, decadence, position=False):
        distance = []
        if point1 is not None and point2 is not None:
            line = np.linalg.norm(point1 - point2)
            for i in range(self.food_feat.shape[0]):
                point = np.array([self.food_feat[i][9], self.food_feat[i][8]])
                vec1 = point1 - point
                vec2 = point2 - point
                distance.append(np.abs(np.cross(vec1, vec2)) / line)
        elif point1 is not None and point2 is None:
            for i in range(self.food_feat.shape[0]):
                distance.append(math.dist(point1, np.array([self.food_feat[i][9], self.food_feat[i][8]])))
        else:
            return None
        sorted_id = sorted(range(len(distance)), key=lambda k: distance[k], reverse=False)
        # 筛选离两个景点最近的前20个餐厅，符合类别要求
        food_cand = []
        comment = []
        cand_count = 0
        for each in sorted_id:
            satisfy = True
            # 鼓浪屿上只能吃鼓浪屿地区的餐厅
            if position and self.food_feat[each][10] != 10001:
                satisfy = False
            # 考虑类型
            for c in self.constraint['lunch-no']:
                if self.food_feat[each][4] == c:
                    satisfy = False
                    break
            # 餐厅不能重复
            for c in self.constraint['select-food']:
                if self.food_feat[each][0] == c:
                    satisfy = False
                    break
            # 满足预算（必须剩一点预算，保证之后几天有饭吃）
            if self.food_feat[each][6] > (self.constraint['user-budget'] - self.constraint['all-budget']) * decadence:
                satisfy = False
            if satisfy:
                food_cand.append(self.food_feat[each])
                comment.append(self.food_feat[each][3])
                cand_count += 1
                if cand_count == 20:
                    break
        # 综合距离2、评分5、评论人数3确定优先顺序
        score = []
        comment_min = min(comment)
        k1 = 30 / (max(comment) - comment_min)
        for i in range(20):
            score.append((20 - 0.5 * i) + food_cand[i][2] * 10 + k1 * (food_cand[i][3] - comment_min))
        sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
        food_choose = food_cand[sorted_id[0]]
        return food_choose

    def improve_plan(self):
        popular = []
        for key in self.plan:
            p = self.plan[key]
            # ========== 插入午餐
            lunch_index = 0
            lunch_min = 99
            pop = 0
            for i in range(len(p)):
                pop += self.spot_feat[p[i][0]-10001][4]
                if abs(p[i][2] - 12) <= lunch_min:
                    lunch_index = i
                    lunch_min = abs(p[i][2] - 12)
            popular.append(pop)
            point1 = point2 = None
            # 12-13 18-19 在index之前插入餐厅
            if 0 <= lunch_min <= 1:
                point1 = np.array([self.spot_feat[p[lunch_index - 1][0] - 10001][6], self.spot_feat[p[lunch_index - 1][0] - 10001][7]])
                point2 = np.array([self.spot_feat[p[lunch_index][0] - 10001][6], self.spot_feat[p[lunch_index][0] - 10001][7]])
            # 在这之外的情况 中间是否能插入
            else:
                point1 = np.array([self.spot_feat[p[lunch_index][0] - 10001][6], self.spot_feat[p[lunch_index][0] - 10001][7]])
            # 考虑类别，排除203冰激淋 219酒吧 220居酒屋 221咖啡店 228零食 232 233面包 250卤味 256西式快餐 257甜点 260小吃 265饮品
            if self.spot_feat[p[lunch_index - 1][0] - 10001][0] == 10001 or self.spot_feat[p[lunch_index - 1][0] - 10001][10] == 10001 or self.spot_feat[p[lunch_index][0] - 10001][0] == 10001 or self.spot_feat[p[lunch_index][0] - 10001][10] == 10001:
                food_choose = self.recommend_food(point1, point2, (1 - (len(self.plan) - key) / len(self.plan)), True)
            else:
                food_choose = self.recommend_food(point1, point2, (1 - (len(self.plan) - key) / len(self.plan)))
            self.constraint['select-food'].append(food_choose[0])
            # 食物: id, name, score, cat, price
            self.plan[key].insert(lunch_index, [food_choose[0], food_choose[1], food_choose[2], food_choose[5], food_choose[6]])
            self.constraint['all-budget'] += food_choose[6]
            for i in range(lunch_index + 1, len(self.plan[key])):
                self.plan[key][i][2] += 1
                self.plan[key][i][3] += 1

            # ========== 插入晚餐
            dinner_index = 0
            dinner_min = 99
            p = self.plan[key]
            for i in range(len(p)):
                if (int(p[i][0] / 10000) == 1) and abs(p[i][3] - 18.5) <= dinner_min:
                    dinner_index = i
                    dinner_min = abs(p[i][3] - 18.5)
            point1 = point2 = None
            if 0 <= dinner_min <= 1 and dinner_index != (len(p) - 1):
                point1 = np.array([self.spot_feat[p[dinner_index][0] - 10001][6], self.spot_feat[p[dinner_index][0] - 10001][7]])
                point2 = np.array([self.spot_feat[p[dinner_index + 1][0] - 10001][6], self.spot_feat[p[dinner_index + 1][0] - 10001][7]])
            else:
                point1 = np.array([self.spot_feat[p[dinner_index][0] - 10001][6], self.spot_feat[p[dinner_index][0] - 10001][7]])
            food_choose = self.recommend_food(point1, point2, (1 - (len(self.plan) - key) / len(self.plan) / 2))
            self.constraint['select-food'].append(food_choose[0])
            self.plan[key].insert(dinner_index + 1, [food_choose[0], food_choose[1], food_choose[2], food_choose[5], food_choose[6]])
            self.constraint['all-budget'] += food_choose[6]
            for i in range(dinner_index + 2, len(self.plan[key])):
                self.plan[key][i][2] += 1
                self.plan[key][i][3] += 1

        # ========== 调整行程，让更受欢迎的行程调到前面
        sorted_id = sorted(range(len(popular)), key=lambda k: popular[k], reverse=True)
        new_plan = {}
        for i in range(len(sorted_id)):
            new_plan[i+1] = self.plan[sorted_id[i]+1]
        self.plan = new_plan

        plan_time = (len(self.plan) - 1) * 24
        if int(list(self.plan.items())[-1][1][-1][0] / 10000) == 1:
            plan_time += list(self.plan.items())[-1][1][-1][-2]
        else:
            plan_time += list(self.plan.items())[-1][1][-2][-2]
        self.constraint['all-time'] = plan_time
        plan_fee = 0
        for key in self.plan:
            for each in self.plan[key]:
                plan_fee += each[4]
        self.constraint['all-budget'] = plan_fee

    def get_trans(self):
        for key in self.plan:
            p = self.plan[key]
            s_lat = self.spot_feat[p[0][0]-10001][6]
            s_long = self.spot_feat[p[0][0]-10001][7]
            s_boat = False
            if p[0][0] == 10001 or self.spot_feat[p[0][0]-10001][10] == 10001:
                s_boat = True
            for i in range(1, len(p)):
                if int(p[i][0] / 10000) == 1:
                    t_lat = self.spot_feat[p[i][0]-10001][6]
                    t_long = self.spot_feat[p[i][0]-10001][7]
                    t_boat = False
                    if p[i][0] == 10001 or self.spot_feat[p[i][0] - 10001][10] == 10001:
                        t_boat = True
                    user_key = '5d5ca7f300800fb6dbefd46be33986ea'
                    origin = str(s_long) + ',' + str(s_lat)
                    destination = str(t_long) + ',' + str(t_lat)
                    output = 'json'
                    city = '0592'
                    # 获取步行方案
                    url = f'https://restapi.amap.com/v5/direction/walking?key={user_key}&origin={origin}&destination={destination}&output={output}&city={city}'
                    walk_txt = json.loads(requests.get(url).text)
                    self.trans['walk'].append([int(walk_txt['route']['paths'][0]['distance']), int(walk_txt['route']['paths'][0]['cost']['duration'])/60])
                    if (s_boat and t_boat) or int(walk_txt['route']['paths'][0]['cost']['duration'])/60 <= 15:
                        self.trans['taxi'].append([0, 0, 0])
                        self.trans['bus'].append([])
                        s_lat = t_lat
                        s_long = t_long
                        s_boat = t_boat
                        continue
                    # 获取打车方案
                    url = f'https://restapi.amap.com/v5/direction/driving?key={user_key}&origin={origin}&destination={destination}&output={output}&city={city}'
                    taxi_txt = json.loads(requests.get(url).text)
                    self.trans['taxi'].append([int(taxi_txt['route']['paths'][0]['distance']), int(taxi_txt['route']['paths'][0]['distance'])/360, int(taxi_txt['route']['taxi_cost'])])
                    # 获取公交方案
                    url = f'https://restapi.amap.com/v5/direction/transit/integrated?key={user_key}&origin={origin}&destination={destination}&output={output}&city1={city}&city2={city}'
                    bus_txt = json.loads(requests.get(url).text)
                    if len(bus_txt['route']['transits']) != 0:
                        self.trans['bus'].append(bus_txt['route']['transits'][0]['segments'])
                    else:
                        self.trans['bus'].append([])
                    s_lat = t_lat
                    s_long = t_long
                    s_boat = t_boat

    def print_trans(self):
        index = 0
        for key in self.plan:
            print('>>> Day', key)
            p = self.plan[key]
            start = p[0]
            # walk: [distance, time]
            # taxi: [distance, time, cost]
            # bus: [{'walking':{}, 'bus': {}}]
            for each in p[1:]:
                if int(each[0] / 10000) == 1:
                    terminal = each
                    print('From', start[1], 'to', terminal[1])
                    walk_time = self.trans['walk'][index][1]
                    # 走路很近，或者起点终点都在岛上
                    if walk_time <= 15 or ((start[0] == 10001 or self.spot_feat[start[0]-10001][10] == 10001) and (terminal[0] == 10001 or self.spot_feat[terminal[0]-10001][10] == 10001)):
                        walk_dis = self.trans['walk'][index][0]
                        if walk_dis > 1000:
                            print('Walk %.2f km for' % (walk_dis / 1000), int(walk_time), 'min')
                        else:
                            print('Walk', walk_dis, 'm for', int(walk_time), 'min')
                    # 用户倾向于公共交通，并且公共交通更方便，或者要往返鼓浪屿
                    elif len(self.trans['bus'][index]) != 0 and ((start[0] == 10001 or self.spot_feat[start[0]-10001][10] == 10001 or terminal[0] == 10001 or self.spot_feat[terminal[0]-10001][10] == 10001) or self.constraint['prefer-trans'] == 0):
                        for seg in self.trans['bus'][index]:
                            if 'walking' in seg:
                                temp_dis = int(seg['walking']['distance'])
                                if temp_dis > 1000:
                                    print('Walk %.2f km for' % (temp_dis / 1000), int(temp_dis / 75), 'min')
                                else:
                                    print('Walk', temp_dis, 'm for', int(temp_dis / 75), 'min')
                            if 'bus' in seg:
                                busline = seg['bus']['buslines'][0]
                                print('From', busline['departure_stop']['name'], 'take', busline['name'], 'to', busline['arrival_stop']['name'])
                                via_stops = busline['via_stops']
                                if len(via_stops) != 0:
                                    for i in range(len(via_stops) - 1):
                                        print(via_stops[i]['name'], '>', end=' ')
                                    print(via_stops[-1]['name'])
                                bus_type = busline['type']
                                temp_dis = int(busline['distance'])
                                speed = 1
                                if bus_type == '普通公交线路':
                                    speed = 30 * 16.7
                                elif bus_type == 'BRT':
                                    speed = 40 * 16.7
                                elif bus_type == '地铁':
                                    speed = 50 * 16.7
                                if bus_type == '轮渡':
                                    print('Estimated time: 15 min')
                                else:
                                    print('Estimated time:', int(temp_dis / speed), 'min')
                    # 用户倾向于打的，或者打的更方便
                    else:
                        print('Take taxi for', end=' ')
                        taxi_dis = self.trans['taxi'][index][0]
                        if taxi_dis > 1000:
                            print('%.2f km' % (taxi_dis / 1000))
                        else:
                            print(taxi_dis, 'm')
                        print('Estimated total time:', self.trans['taxi'][index][1], 'min')
                        print('Estimated total fee:', self.trans['taxi'][index][2], 'rmb')
                    print('-----')
                    start = terminal
                    index += 1

    # 根据POI种类丰富度和行程时间安排评估行程分数
    def evaluate(self):
        cat = set()
        spot_time = 0
        play_time = 0
        for key in self.plan:
            p = self.plan[key]
            if int(p[-1][0] / 10000) == 1:
                play_time += (p[-1][-2] - p[0][2])
            else:
                play_time += (p[-2][-2] - p[0][2])
            for each in p:
                if int(each[0] / 10000) == 1:
                    cat.add(self.spot_feat[each[0]-10001][2])
                    spot_time += (each[3] - each[2])
        self.score = 0.5 * len(cat) / 6 + 0.3 * spot_time / play_time + 0.2 * play_time / self.constraint['user-time'] * 2
        print('Score: %.2f' % (self.score * 100), end='')
        print('/100')


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
    cal_attraction()
    # 1鼓浪屿，3园林植物园，5日光岩，6环岛路，7曾厝垵，8海底世界，9中山路，36海湾公园，41厦大
    # 1-5-8-9-6-7-41-3-36
    cost = get_cost()
    plan = ant_colony(cost, [10001, 10003, 10005, 10006, 10007, 10008, 10009, 10036, 10041])
    print('---------- Original Plan ----------')
    print_plan(plan)
    score = evaluate(plan)
    '''
    plan = {1: [[10001, '鼓浪屿', 9, 16.0, 0.0], [10008, '厦门海底世界', 9, 12.0, 107.0],
                [21160, '那私厨｜红砖老别墅餐厅(鼓浪屿店)', 4.9, '闽菜', 100], [10005, '日光岩', 13.0, 15.0, 50.0],
                [10009, '中山路步行街', 17.5, 20.5, 0.0], [21407, '醉得意(厦禾店)', 4.8, '其他', 55]],
            2: [[10041, '厦门大学', 9, 11.0, 0.0], [21192, '金家港海鲜大排档(中山路店)', 4.9, '海鲜', 145],
                [10007, '曾厝垵', 12.5, 15.5, 0.0], [10002, '胡里山炮台', 16.0, 18.0, 23.0],
                [21887, '上青本港海鲜(后江埭路店)', 4.7, '海鲜', 204], [10115, '白鹭洲公园', 20.0, 22.0, 0.0]]}
    constraint = {'user-time': 48,
                  'user-budget': 1000,
                  'all-time': 0,
                  'all-budget': 0,
                  'prefer-trans': 0,
                  'select-spot': [10001, 10002, 10003, 10005, 10006, 10007, 10008, 10009, 10041],
                  'select-food': [],
                  'lunch-no': [203, 219, 220, 221, 228, 232, 233, 250, 256, 257, 260, 265],
                  'dinner-no': [203, 221, 228, 232, 233, 250, 256, 257, 260, 265]}
    plan_manager = PlanManager(constraint)
    plan_manager.plan = plan
    plan_manager.get_trans()
    plan_manager.print_trans()

