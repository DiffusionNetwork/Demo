# encoding:utf-8
import numpy as np
import os
import collections


class m_network:

    network_name = ""
    base_addr = ""
    node_num = 0
    edge_num = -1
    probability = 1
    category = -1
    graph = None
    edge = None

    def __init__(self, node_num, probability=1.0, base_addr="", network_name=""):
        self.base_addr = base_addr
        self.network_name = network_name
        self.node_num = node_num
        self.graph = np.zeros([node_num, node_num]) # use 1 for edge, 0 for no edge (directed)
        self.set_edge_para(probability)

    def set_edge_para(self, probability):
        assert (1 >= probability >= 0)
        self.probability = probability
        self.edge = self.graph * probability

    def read_network(self):
        if self.network_name == "":
            print("This network is not readable")
            exit(1)
        if '.txt' not in self.network_name:
            self.network_name += '.txt'
        f = open(self.base_addr + self.network_name, 'r')
        t = f.readline()
        while t:
            arr = t.rstrip('\n')
            arr = arr.split('\t')
            nd1 = int(arr[0]) - 1
            nd2 = int(arr[1]) - 1
            self.graph[nd1, nd2] = 1
            self.edge[nd1, nd2] = self.probability
            t = f.readline()
        f.close()

    def add_edge(self, nd1, nd2): # between 0 and node_num -1
        self.graph[nd1, nd2] = 1
        self.edge[nd1, nd2] = self.probability
        self.edge_num += 1

    def del_edge(self, nd1, nd2):
        self.graph[nd1, nd2] = 0
        self.edge[nd1, nd2] = 0

    def get_edge(self, nd1, nd2):
        return self.edge[nd1, nd2]

    def get_edge_m(self):
        return self.edge

    def get_graph_m(self):
        return self.graph

    def clear_graph(self):   
        self.graph = np.zeros([self.node_num, self.node_num])
        self.edge = np.zeros([self.node_num, self.node_num])
        self.edge_num = 0

    def overwrite_graph(self, graph):
        self.graph = graph

    def overwrite_edge(self, edge):
        self.edge = np.copy(edge)

    def write_network(self, url):
        g = open(url + self.network_name, 'w')
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.graph[i, j] > 0:
                    g.write(str(i+1) + "\t" + str(j+1) + "\n")
        g.flush()
        g.close()

    def display_edges(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.graph[i, j] > 0:
                    print("edge [%d, %d] prob = %f" % (i, j, self.edge[i, j]))


class m_cascade:

    cascade_name = ""
    base_addr = ""
    node_num = 0
    cascade_num = 0
    cascade = None

    cat_num = 0
    labels = None
    ground_truth_labels = None
    mi_matrix = None

    def __init__(self, node_num, cascade_num, base_addr="", cascade_name="", cat_num=1):
        self.base_addr = base_addr
        self.cascade_name = cascade_name
        self.node_num = node_num
        self.cascade_num = cascade_num
        self.cascade = np.zeros([cascade_num, node_num]) # use 1 for infected, 0 for not infected (directed)
        self.cat_num = cat_num
        # self.cat = np.zeros([cascade_num])
        self.init_cat_labels()
        self.mi_matrix = np.zeros([cat_num, node_num, node_num])

    def init_cat_labels(self):
        if self.cat_num <= 1:
            self.labels = np.zeros([self.cascade_num])
        else:
            self.labels = np.random.randint(self.cat_num, size=self.cascade_num)  # 随机初始化aspect标签

    def read_cascade(self):
        if self.cascade_name == "":
            print("This cascade is not readable")
            exit(1)
        if '.txt' not in self.cascade_name:
            self.cascade_name += '.txt'
        f = open(self.base_addr + self.cascade_name, 'r')
        t = f.readline()
        cascade_cnt = 0
        while t:
            arr = t.rstrip('\n')
            arr = arr.split('\t')   # size: node_num
            for i in range(self.node_num):
                if arr[i] == "1":
                    self.cascade[cascade_cnt][i] = 1  # 组合后感染状态结果
            cascade_cnt += 1
            t = f.readline()
        f.close()
        if cascade_cnt != self.cascade_num:
            print("cascade number not match")
            exit(1)

    def get_cascade(self):
        return self.cascade

    def get_status(self, cascade_id, nd_id):
        return self.cascade[cascade_id, nd_id]

    def write_cascade(self, url):
        g = open(url + self.cascade_name, 'w')
        for cascade_idx in range(self.cascade_num):
            for i in range(self.node_num):
                g.write(str(int(self.cascade[cascade_idx, i])) + "\t")
            g.write("\n")
        g.flush()
        g.close()

    def set_cascade(self, cascade):
        if len(cascade[0, :]) != self.node_num or len(cascade[:,0]) != self.cascade_num:
            print("cascade matrix size does not match")
            exit(1)
        self.cascade = cascade

    def count_exist(self, node_id1, state1, node_id2, state2, dest_label):
        cnt = 0
        for i in range(self.cascade_num):
            if self.labels[i] == dest_label and self.cascade[i][node_id1] == state1 and self.cascade[i][node_id2] == state2:
                cnt += 1
        return cnt

    def label_renew(self, labels):
        self.labels = np.copy(labels)

    def prune_with_label(self, label):  # 从组合图中取出属于aspect=label的记录 --> 返回selected_cas
        counter = collections.Counter(self.labels) # labels计数：各个aspect label的个数
        match_num = counter[label]  # aspect=label 的个数
        idx = 0
        selected_cas = np.zeros([match_num, self.node_num])
        for a in range(self.cascade_num):
            if self.labels[a] == label:
                selected_cas[idx] = self.cascade[a]
                idx += 1
        return selected_cas, match_num

    def init_ground_truth_labels(self, ground_truth_labels):
        # print(ground_truth_labels)
        self.ground_truth_labels = ground_truth_labels

    def cal_labels_accuracy(self):
        # print(self.labels)
        # print(self.ground_truth_labels)
        # os.system("pause")
        tmp = self.labels - self.ground_truth_labels
        zero_counter = collections.Counter(tmp)
        match_num = zero_counter[0]
        return match_num / self.cascade_num

    def cal_mi_for_all_cas(self):
        for cat in range(self.cat_num):
            pruned_cas, pruned_cas_num = self.prune_with_label(cat)
            self.mi_matrix[cat] = cal_mi_from_cascade_full(pruned_cas, pruned_cas_num, self.node_num)


def generate_forward_cascade(net, cascade_num, init_rate, timestamp=0):
    # source_type indicates the way to choose source nodes
    # default = random source
    if not isinstance(net, m_network):
        print("in generate_forward_cascade(net, cascade_num) \ninput \"net\" should be the instance of m_network")
        exit(1)
    cascade = np.zeros([cascade_num, net.node_num])
    for cascade_id in range(cascade_num):
        status = np.zeros([net.node_num])
        valid_node = np.zeros([net.node_num])
        source_id = np.random.randint(net.node_num, size=int(init_rate*net.node_num))
        valid_node[source_id] = 1
        time = 1 # next time stamp
        while sum(valid_node) > 0:
            status = status + valid_node
            waiting_list = np.argwhere(valid_node >= 1).flatten()
            # print(waiting_list )
            # print(sum(valid_node))
            print("time = " + str(time) + " , with activated = " + str(sum(valid_node)))
            valid_node = valid_node * 0
            time += 1
            for idx in waiting_list:
                edge = net.edge[idx, :] # net.graph[idx, :] * net.edge[idx, :]
                # print(np.argwhere((0 < edge)&(edge < 1)))
                rand_para = np.random.random(net.node_num)
                active_node_idx = np.argwhere((edge > 0) & (edge >= rand_para) & (status <= 0)).flatten()
                if timestamp:
                    valid_node[active_node_idx] = time
                else:
                    valid_node[active_node_idx] = 1
                # print(edge[active_node_idx])
        cascade[cascade_id, :] += status
        # print(sum(status))
        # print("finish " + str(cascade_id) )
        exit(10)
    return cascade


def cal_mi_from_cascade(cascade_mat, cas_num, node_num): #计算互信息矩阵
    mi_matrix = np.zeros([node_num, node_num])
    for i in range(node_num):
        for j in range(i):
            pos_cnt = 0
            cnt1 = cnt2 = 0
            for cas in range(cas_num):
                if cascade_mat[cas, i] == 1 and cascade_mat[cas, j] == 1:
                    pos_cnt += 1
                    cnt1 += 1
                    cnt2 += 1
                elif cascade_mat[cas, i] == 1:
                    cnt1 += 1
                elif cascade_mat[cas, j] == 1:
                    cnt2 += 1
            if pos_cnt * cnt1 * cnt2 == 0:
                mi = 0
            else:
                mi = (pos_cnt/cas_num)*np.log((pos_cnt/cas_num)/((cnt1/cas_num)*(cnt2/cas_num)))
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix


def cal_mi_from_cascade_full(cascade_mat, cas_num, node_num):
    mi_matrix = np.zeros([node_num, node_num])
    for i in range(node_num):
        for j in range(i):
            cnt_1_1 = cnt_0_1 = cnt_1_0 = cnt_0_0 = 0
            for cas in range(cas_num):
                if cascade_mat[cas, i] == 1 and cascade_mat[cas, j] == 1:
                    cnt_1_1 += 1
                elif cascade_mat[cas, i] == 1 and cascade_mat[cas, j] == 0:
                    cnt_1_0 += 1
                elif cascade_mat[cas, i] == 0 and cascade_mat[cas, j] == 1:
                    cnt_0_1 += 1
                elif cascade_mat[cas, i] == 0 and cascade_mat[cas, j] == 0:
                    cnt_0_0 += 1
            mi = 0
            if cnt_0_0 * (cnt_0_0 + cnt_0_1) * (cnt_0_0 + cnt_1_0) > 0:
                mi += (cnt_0_0 / cas_num) * np.log((cnt_0_0 / cas_num) / (((cnt_0_0 + cnt_0_1)/cas_num) * ((cnt_0_0 + cnt_1_0)/cas_num)))
            if cnt_1_0 * (cnt_1_0 + cnt_1_1) * (cnt_0_0 + cnt_1_0) > 0:
                mi += (cnt_1_0 / cas_num) * np.log((cnt_1_0 / cas_num) / (((cnt_1_0 + cnt_1_1)/cas_num) * ((cnt_0_0 + cnt_1_0)/cas_num)))
            if cnt_0_1 * (cnt_0_0 + cnt_0_1) * (cnt_1_1 + cnt_0_1) > 0:
                mi += (cnt_0_1 / cas_num) * np.log((cnt_0_1 / cas_num) / (((cnt_0_0 + cnt_0_1)/cas_num) * ((cnt_1_1 + cnt_0_1)/cas_num)))
            if cnt_1_1 * (cnt_1_0 + cnt_1_1) * (cnt_0_1 + cnt_1_1) > 0:
                mi += (cnt_1_1 / cas_num) * np.log((cnt_1_1 / cas_num) / (((cnt_1_0 + cnt_1_1)/cas_num) * ((cnt_0_1 + cnt_1_1)/cas_num)))

            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix
