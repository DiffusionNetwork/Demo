# encoding:utf-8

# Idea from 多重传播类型问题思路 -> 3
import collections

import LMDN_network as nt
import numpy as np
import math
from time import time
import os
from sklearn.cluster import KMeans
from collections import Counter
import time
from itertools import combinations


# fast sub-network topology reconstruction using edge-assignment
def g_edge_score(parent_node_id1, child_node_id2, cascade, dest_label):
    score = 0.0
    # calculate the source when parent node was infected
    parent_state = 1
    N_i_j_0 = cascade.count_exist(parent_node_id1, parent_state, child_node_id2, 0, dest_label)
    N_i_j_1 = cascade.count_exist(parent_node_id1, parent_state, child_node_id2, 1, dest_label)
    score += math.log(math.factorial(N_i_j_0)) + math.log(math.factorial(N_i_j_1)) - math.log(
        math.factorial(N_i_j_0 + N_i_j_1 + 1))

    # calculate the source when parent node was NOT infected
    parent_state = 0
    N_i_j_0 = cascade.count_exist(parent_node_id1, parent_state, child_node_id2, 0, dest_label)
    N_i_j_1 = cascade.count_exist(parent_node_id1, parent_state, child_node_id2, 1, dest_label)
    score += math.log(math.factorial(N_i_j_0)) + math.log(math.factorial(N_i_j_1)) - math.log(
        math.factorial(N_i_j_0 + N_i_j_1 + 1))
    return score


def g_parent_score(pruned_cascade_mat, child_node_id, parent_node_list, node_num):
    # parent_num = len(parent_node_list)
    # parent_state_num = int(math.pow(2, parent_num))
    # full_state = np.zeros([node_num])
    # score = 0
    # for i in range(parent_state_num):
    #     # iterate time: parent_state_num
    #     state_str = bin(i)
    #     state_str = state_str.strip('0b')
    #     full_state *= 0
    #     for j in range(len(state_str)):
    #         if state_str[len(state_str) - j - 1] == '1':
    #             full_state[parent_node_list[parent_num - j - 1]] = 1
    #     # for j in range(cas_num):
    #     full_state[child_node_id] = 0
    #     cnt_arr = np.sum((pruned_cascade_mat & full_state) ^ full_state, axis=1).flatten()
    #     cnt0_cnter = collections.Counter(cnt_arr)
    #     cnt0 = cnt0_cnter[0]
    #
    #     full_state[child_node_id] = 1
    #     cnt_arr = np.sum((pruned_cascade_mat & full_state) ^ full_state, axis=1).flatten()
    #     cnt0_cnter = collections.Counter(cnt_arr)
    #     cnt1 = cnt0_cnter[0]
    #
    #     score += np.log(np.math.factorial(cnt0)) + np.log(np.math.factorial(cnt1) - np.log(np.math.factorial(cnt1 + cnt0 + 1)))
    # return score
    inner_parent_list = parent_node_list.copy()
    inner_parent_list.append(child_node_id)
    cmp_parent_with_child_np = np.array(inner_parent_list)
    # print(cmp_parent_with_child_np)
    parent_num = len(parent_node_list)
    parent_state_num = int(math.pow(2, parent_num))   # 父节点可以有parent_state_num种状态组合
    parent_and_child_states = np.zeros([node_num])

    score = 0
    for i in range(parent_state_num):
        state_str = bin(i)
        # state_str = state_str.strip('0b')  # 两头都去了，不对
        state_str = state_str.lstrip('0b')
        parent_and_child_states *= 0
        for j in range(len(state_str)):   # 改变父节点的状态     ？父节点状态没有变
            # if parent_and_child_states[len(state_str) - j - 1] == '1':  # 永远不可能是1，不对
            if state_str[len(state_str) - j - 1] == '1':
                parent_and_child_states[parent_node_list[parent_num - j - 1]] = 1

        parent_and_child_states[child_node_id] = 0

        cmp_full_cas = pruned_cascade_mat[:, cmp_parent_with_child_np.astype(int)]
        cmp_dest_pattern = parent_and_child_states[cmp_parent_with_child_np.astype(int)].flatten()

        res = np.sum(cmp_full_cas == cmp_dest_pattern, axis=1)
        cnter = collections.Counter(res)
        cnt0 = cnter[len(cmp_dest_pattern)]

        parent_and_child_states[child_node_id] = 1
        cmp_dest_pattern = parent_and_child_states[cmp_parent_with_child_np.astype(int)].flatten()

        res = np.sum(cmp_full_cas == cmp_dest_pattern, axis=1)
        cnter = collections.Counter(res)
        cnt1 = cnter[len(cmp_dest_pattern)]
        # print(cnt1)
        # print(cnt0)
        score += np.log(float(np.math.factorial(cnt0))) + np.log(float(np.math.factorial(cnt1))) - np.log(float(np.math.factorial(cnt1 + cnt0 + 1)))
    return score





"""
求父节点数量上限
"""
def cal_upper_bound(pruned_cascade_mat, node_num, cas_num, parent_node_list):
    # for i in range(node_num):
    parent_num = len(parent_node_list)
    total_combination = int(math.pow(2, parent_num))
    tmp_cas = pruned_cascade_mat[:, parent_node_list]
    # tmp_cas_str = []
    # for i in range(len(tmp_cas)):  # 转为str
    #     tmp_cas_str.append(''.join(str(int(s)) for s in tmp_cas[i]))
    # parent_states = collections.Counter(tmp_cas_str)
    # parent_states_num = len(list(parent_states))
    
    tmp_cnt_array = np.zeros([total_combination])  # 太大了，运行不了
    tmp_cas_token = np.zeros([cas_num])
    # print(tmp_cas)
    for t in range(parent_num):
        tmp_cas_token += np.power(2, t) * tmp_cas[:, t]
    for cas_id in range(cas_num):
        # print(tmp_cas_token)
        tmp_cnt_array[int(tmp_cas_token[cas_id])] = 1
    non_exist_cnt = total_combination - np.sum(tmp_cnt_array)
    # non_exist_cnt = total_combination - parent_states_num
    # upper_limit = np.log(np.log(cas_num + 1) * np.log(np.e * (cas_num + 1) / 2) + non_exist_cnt)
    upper_limit = np.log((cas_num + 1) * np.log(np.e * (cas_num + 1) / 2) + non_exist_cnt)
    return np.ceil(upper_limit)  # 取上整


# 速度缓慢，必须使用矩阵运算，故改为 net_diffusion_rate_5a_m，此版本函数放弃
# def net_diffusion_rate_5a(cascade, dest_label, sub_network_structure, display=0):
#     # network diffusion rate reconstruction
#     # using 5A
#     p_matrix = np.random.random([cascade.node_num, cascade.node_num])
#     f = np.zeros([cascade.cascade_num, cascade.node_num, cascade.node_num])
#     sub_network_structure.graph[sub_network_structure.graph > 0] = 1
#     graph = sub_network_structure.graph
#     max_err = 0.0001
#     iter_cnt = 0
#     while True:
#         iter_cnt += 1
#         # first: update f
#         for idx in range(cascade.cascade_num):
#             if cascade.labels[idx] != dest_label:
#                 continue
#             for i in range(cascade.node_num):
#                 if cascade.cascade[idx, i] == 0:
#                     f[idx, i] = np.zeros([cascade.node_num])
#                     continue
#                 for j in range(cascade.node_num):
#                     if cascade.cascade[idx, j] == 0:
#                         f[idx, i, j] = 0
#                     else:
#                         sub_internal = 1
#                         for par in range(cascade.node_num):
#                             if graph[par, j] > 0 and cascade.cascade[idx, par] > 0:
#                                 sub_internal *= (1 - p_matrix[par, j])
#                         if sub_internal >= 1:
#                             f[idx, i, j] = 0
#                         else:
#                             f[idx, i, j] = p_matrix[i, j] / (1 - sub_internal)
#         # second: update p
#         finish_label = True
#         for i in range(cascade.node_num):
#             for j in range(cascade.node_num):
#                 cnt = 0
#                 sum = 0
#                 for cas in range(cascade.cascade_num):
#                     if cascade.labels[cas] == dest_label:
#                         if cascade.cascade[cas, i] == 1 and cascade.cascade[cas, j] == 1:
#                             sum += f[cas][i][j]
#                         if cascade.cascade[cas, i] == 0 and cascade.cascade[cas, j] == 1:
#                             cnt += 1
#                 if sum + cnt != 0:
#                     assign_value = sum/(sum+cnt)
#                 else:
#                     assign_value = 0
#                 if finish_label and np.abs(assign_value - p_matrix[i, j]) > max_err:
#                     finish_label = False
#                 p_matrix[i, j] = assign_value
#         # third: judge when to stop
#         if finish_label:
#             sub_network_structure.edge = p_matrix
#             if display == 1:
#                 print("finish EM process")
#             break
#         elif display == 1:
#             print("iterate %d times;" % iter_cnt)
#             break
#     return p_matrix


# 使用矩阵运算版本的net_diffusion_rate_5a_m，取代net_diffusion_rate_5a
# 这个函数更新p的方式跟文档中写的不一样
'''
求权
sub_network_structure:某个aspect的结构，即边 <--- sub_net_list[aspect]
'''
def net_diffusion_rate_5a_m(cascade, dest_label, sub_network_structure, display=0):
    # network diffusion rate reconstruction
    # using 5A

    sub_network_structure.graph[sub_network_structure.graph > 0] = 1
    graph = sub_network_structure.graph
    graph[graph > 0] = 1
    dest_cascade, cascade_num = cascade.prune_with_label(dest_label)
    node_num = cascade.node_num
    f = np.zeros([cascade_num, node_num, node_num])
    f_m = np.zeros([cascade_num, node_num, node_num])
    p_matrix = np.random.random([node_num, node_num])
    max_err = 0.0001
    iter_cnt = 0
    neg_cnt = np.zeros([node_num, node_num])
    # start pre-calculation
    node_positive_cascade_mask = np.zeros([node_num, node_num, cascade_num])
    for i in range(node_num):
        for j in range(node_num):
            if graph[i, j] <= 0:
                continue
            for cas in range(cascade_num):
                if dest_cascade[cas, i] == 1 and dest_cascade[cas, j] == 1:
                    node_positive_cascade_mask[i, j, cas] = 1
                elif dest_cascade[cas, i] != dest_cascade[cas, j]:
                    neg_cnt[i, j] += 1
    # end pre-calculation

    while True:
        iter_cnt += 1
        # first: update f
        tmp_1_p = 1 - p_matrix
        clock1 = time.time()
        for idx in range(cascade_num):
            # 1-multi(1-p)
            # graph_cas = graph * dest_cascade[idx].reshape([node_num, 1])
            # tmp = tmp_1_p * graph_cas
            # tmp[tmp == 0] = 1
            # sub = 1 - np.prod(tmp, axis=0)
            # sub[sub == 0] = np.inf
            # f[idx, :, :] = np.copy(p_matrix / sub)
            # f[idx, :, :] = f[idx, :, :] * dest_cascade[idx]
            # f[idx, :, :] = f[idx, :, :] * dest_cascade[idx].reshape([node_num, 1])

            # sum p
            graph_cas = graph * dest_cascade[idx].reshape([node_num, 1])
            tmp = p_matrix * graph_cas
            sum_p = np.sum(tmp, axis = 0)
            sum_p[sum_p == 0] = np.inf
            f[idx, :, :] = np.copy(p_matrix / sum_p)
            f[idx, :, :] = f[idx, :, :] * dest_cascade[idx]
            f[idx, :, :] = f[idx, :, :] * dest_cascade[idx].reshape([node_num, 1])


            # for i in range(node_num):
            #     for j in range(node_num):
            #         inner_f = 1
            #         if dest_cascade[idx, i] <= 0 or dest_cascade[idx, j] <= 0:
            #             f_m[idx, i, j] = 0
            #             continue
            #         for par in range(node_num):
            #             if graph[par, j] > 0 and dest_cascade[idx, par] > 0:
            #                 inner_f *= (1 - p_matrix[par, j])
            #         inner_f = 1 - inner_f
            #         if inner_f == 0:
            #             f_m[idx, i, j] = 0
            #         else:
            #             f_m[idx, i, j] = p_matrix[i, j] / inner_f

        # second: update p
        finish_label = True
        clock2 = time.time()
        for i in range(node_num):
            for j in range(node_num):
                if graph[i, j] <= 0:
                    assign_value = 0
                else:
                    # pos_sum = np.sum(f[np.argwhere(node_positive_cascade_mask[i, j] == 1).flatten(), i, j])
                    pos_sum = np.sum(f[:, i, j] * node_positive_cascade_mask[i, j])
                    if pos_sum + neg_cnt[i, j] != 0:
                        assign_value = pos_sum / (pos_sum + neg_cnt[i, j])
                        # print("none - zero assign %f" % assign_value)
                    else:
                        assign_value = 0

                # pos_sum_inner = 0
                # neg_cnt_inner = 0
                # if graph[i, j] <= 0:
                #     assign_value = 0
                # else:
                #     for cas in range(cascade_num):
                #         if dest_cascade[cas, i] == 1 and dest_cascade[cas, j] == 1:
                #             # node_positive_cascade_mask[i, j, cas] = 1
                #             pos_sum_inner += f[cas, i, j]
                #         elif dest_cascade[cas, i] == 0 and dest_cascade[cas, j] == 1:
                #             neg_cnt_inner += 1
                #         elif dest_cascade[cas, i] == 1 and dest_cascade[cas, j] == 0:
                #             neg_cnt_inner += 1
                #
                #     if pos_sum_inner + neg_cnt_inner != 0:
                #         assign_value = pos_sum_inner/(pos_sum_inner + neg_cnt_inner)
                #         # print("none - zero assign %f" % assign_value)
                #     else:
                #         assign_value = 0

                if finish_label and np.abs(assign_value - p_matrix[i, j]) > max_err:
                    finish_label = False
                p_matrix[i, j] = assign_value
                if assign_value > 1:
                    print("assign value = %f" % assign_value)
                    os.system("pause")
        clock3 = time.time()
        # print("time interval 1 = %f; 2 = %f" % (clock2 - clock1, clock3 - clock2))
        # third: judge when to stop
        if finish_label:
            sub_network_structure.edge = p_matrix
            if display == 1:
                print("finish EM process with %d iterations" % iter_cnt)
            break
        elif display == 1:
            print("iterate %d times;" % iter_cnt)
    return p_matrix


# 去除普通EM-5A算法中的父节点要求，直接进行计算，查看是否可行，决定是否可以直接比较Q
def net_diffusion_rate_general_5a_m(cascade, dest_label, sub_network_structure, display=0):
    # network diffusion rate reconstruction
    # using 5A general extension

    sub_network_structure.graph[sub_network_structure.graph > 0] = 1
    graph = sub_network_structure.graph
    graph[graph > 0] = 1
    dest_cascade, cascade_num = cascade.prune_with_label(dest_label)
    node_num = cascade.node_num
    f = np.zeros([cascade_num, node_num, node_num])
    f_m = np.zeros([cascade_num, node_num, node_num])
    p_matrix = np.random.random([node_num, node_num])
    max_err = 0.0001
    iter_cnt = 0
    neg_cnt = np.zeros([node_num, node_num])

    # start pre-calculation
    node_positive_cascade_mask = np.zeros([node_num, node_num, cascade_num])
    for i in range(node_num):
        for j in range(node_num):
            # if graph[i, j] <= 0:
            #     continue
            for cas in range(cascade_num):
                if dest_cascade[cas, i] == 1 and dest_cascade[cas, j] == 1:
                    node_positive_cascade_mask[i, j, cas] = 1
                elif dest_cascade[cas, i] != dest_cascade[cas, j]:
                    neg_cnt[i, j] += 1
    # end pre-calculation
    while True:
        iter_cnt += 1
        # first: update f
        tmp_1_p = 1 - p_matrix
        clock1 = time.time()
        for idx in range(cascade_num):
            # graph_cas = graph * dest_cascade[idx].reshape([node_num, 1])
            graph_cas = dest_cascade[idx].reshape([node_num, 1])
            tmp = tmp_1_p * graph_cas
            tmp[tmp == 0] = 1
            sub = 1 - np.prod(tmp, axis=0)
            sub[sub == 0] = np.inf
            f[idx, :, :] = np.copy(p_matrix / sub)
            f[idx, :, :] = f[idx, :, :] * dest_cascade[idx]
            f[idx, :, :] = f[idx, :, :] * dest_cascade[idx].reshape([node_num, 1])

        # second: update p
        finish_label = True
        clock2 = time.time()
        for i in range(node_num):
            for j in range(node_num):
                # if graph[i, j] <= 0:
                #     assign_value = 0
                # else:
                    # pos_sum = np.sum(f[np.argwhere(node_positive_cascade_mask[i, j] == 1).flatten(), i, j])

                pos_sum = np.sum(f[:, i, j] * node_positive_cascade_mask[i, j])
                if pos_sum + neg_cnt[i, j] != 0:
                    assign_value = pos_sum / (pos_sum + neg_cnt[i, j])
                    # print("none - zero assign %f" % assign_value)
                else:
                    assign_value = 0

                if finish_label and np.abs(assign_value - p_matrix[i, j]) > max_err:
                    finish_label = False
                p_matrix[i, j] = assign_value
                if assign_value > 1:
                    print("assign value = %f" % assign_value)
                    os.system("pause")
        clock3 = time.time()
        # print("time interval 1 = %f; 2 = %f" % (clock2 - clock1, clock3 - clock2))
        # third: judge when to stop
        if finish_label:
            sub_network_structure.edge = p_matrix
            if display == 1:
                print("finish EM process with %d iterations" % iter_cnt)
            break
        elif display == 1:
            print("iterate %d times;" % iter_cnt)
    return p_matrix

'''
求边
dest_label:第几个aspect
'''
def net_parent_set_reconstruct_5a(dest_net, src_cascade, dest_label, display=0):
    cascade, cas_num = src_cascade.prune_with_label(dest_label) #cascade:aspect=dest_label的图结构
    if cas_num == 0:
        return dest_net

    dest_net.clear_graph()
    node_num = src_cascade.node_num
    # upper_bound = np.ceil(math.log((node_num+1) * math.log(math.e * (node_num+1)/2, 2),2))
    upper_bound = 8 # 200个点
    
    cascade, cas_num = src_cascade.prune_with_label(dest_label) 
    # start run the reconstruction
    mi_matrix = nt.cal_mi_from_cascade(cascade, cas_num, node_num)  # 互信息矩阵

    mi_matrix_sorted = np.copy(mi_matrix)
    mi_matrix_sorted.sort()  # 在行上排序
    top_10_list = mi_matrix_sorted[:, -10].reshape(-1, 1)  # 选择 第10大 为界
    mi_matrix[mi_matrix < top_10_list] = 0

    mi_flatten = mi_matrix[np.nonzero(mi_matrix)].reshape(-1, 1) # 非零元素

    class_pred = KMeans(n_clusters=2, random_state=0).fit_predict(mi_flatten)

    class_1_min = np.min(mi_flatten[np.argwhere(class_pred == 1)])
    class_0_min = np.min(mi_flatten[np.argwhere(class_pred == 0)])

    if display == 1:
        print("class_1_max = %f, class_0_max = %f" % (class_1_min, class_0_min))
    if class_1_min > class_0_min:
        threshold = class_1_min
    else:
        threshold = class_0_min
    if display == 1:
        print("threshold = %f" % threshold)
    for i in range(node_num):
        mi_matrix[:, i][mi_matrix[:, i] < threshold] = 0
        parent_num = len(np.nonzero(mi_matrix[:, i])[0])  # np.nonzero返回的是tuple数组
        sort_idx = mi_matrix[:, i].argsort()  # 返回从小到大排序后的索引
        if parent_num > upper_bound:
            parent_id_set = sort_idx[-int(upper_bound):]    # bound个父节点
        elif parent_num == 0:
            parent_id_set = []
        else:
            parent_id_set = sort_idx[-parent_num:]
        for j in range(len(parent_id_set)):
            dest_net.add_edge(parent_id_set[j], i)
        
        # for j in range(node_num):
        #     if mi_matrix[i, j] >= threshold:
        #         dest_net.add_edge(i, j)       # KMeans对边剪枝  这里edge都是1
    
    # for i in range(node_num):
    #     number = 0
    #     nonzero_i = mi_matrix[i][np.nonzero(mi_matrix[i])].reshape(-1, 1)
    #     class_pred = KMeans(n_clusters=2, random_state=0).fit_predict(nonzero_i)
        
    #     class_1_min = np.min(nonzero_i[np.argwhere(class_pred == 1)])
    #     class_0_min = np.min(nonzero_i[np.argwhere(class_pred == 0)])
        
    #     if class_1_min > class_0_min:
    #         threshold = class_1_min
    #     else:
    #         threshold = class_0_min
    #     for j in range(node_num):
    #         if mi_matrix[i][j] >= threshold and number < upper_bound:
    #             dest_net.add_edge(i, j)
    #             number += 1

    return dest_net

# 单质
def TWIND(estimated_graph, cascade, cas_num, node_num):
    upper_bound = 8
    mi_matrix = nt.cal_mi_from_cascade(cascade, cas_num, node_num)  # 互信息矩阵
    
    mi_matrix_sorted = np.copy(mi_matrix)
    mi_matrix_sorted.sort()  # 在行上排序
    top_10_list = mi_matrix_sorted[:, -10].reshape(-1, 1)  # 选择 第10大 为界
    mi_matrix[mi_matrix < top_10_list] = 0

    mi_flatten = mi_matrix[np.nonzero(mi_matrix)].reshape(-1, 1) # 非零元素
    
    # 找出两个最大值，作为KMeans初始点   
    sorted_mi = np.unique(mi_flatten).reshape(-1, 1)   # 去除重复值并排序
    print(sorted_mi)
    print(sorted_mi[-2:])
    # print(np.array([sorted_mi[0], sorted_mi[-1]]))

    class_pred = KMeans(n_clusters=2, init=sorted_mi[-2:], n_init = 1).fit_predict(mi_flatten)

    class_1_min = np.min(mi_flatten[np.argwhere(class_pred == 1)])
    class_0_min = np.min(mi_flatten[np.argwhere(class_pred == 0)])

    if class_1_min > class_0_min:
        threshold = class_1_min
    else:
        threshold = class_0_min
    for i in range(node_num):
        mi_matrix[:, i][mi_matrix[:, i] < threshold] = 0
        parent_num = len(np.nonzero(mi_matrix[:, i])[0])  # np.nonzero返回的是tuple数组
        sort_idx = mi_matrix[:, i].argsort()  # 返回从小到大排序后的索引
        if parent_num > upper_bound:
            parent_id_set = sort_idx[-int(upper_bound):]    # bound个父节点
        elif parent_num == 0:
            parent_id_set = []
        else:
            parent_id_set = sort_idx[-parent_num:]
        for j in range(len(parent_id_set)):
            estimated_graph[parent_id_set[j], i] = 1
    return estimated_graph



# 在计算似然函数的时候跟文档中写的不太一样
def net_cascade_likelihood_5a(cascade_id, cascade, sub_network):
    # this is the likelihood of a cascade to a network
    # using 5A
    infected_likelihood = 0
    uninfected_likelihood = 0
    for i in range(cascade.node_num):
        if cascade.cascade[cascade_id, i] == 1:
            prod = 1
            m_sum = 0
            for t in range(cascade.node_num):
                if sub_network.edge[t, i] > 0:
                    if cascade.cascade[cascade_id, t] > 0:
                        prod *= (1 - sub_network.edge[t, i])
                    else:
                        m_sum += math.log(1 - sub_network.edge[t, i])
            # print("prod = %f " % prod)
            if prod >= 1:
                infected_likelihood += m_sum
            else:
                infected_likelihood += (math.log(1 - prod) + m_sum)
        else:
            m_sum = 0
            for q in range(cascade.node_num):
                if sub_network.edge[q, i] > 0 and cascade.cascade[cascade_id, q] > 0:
                    m_sum += math.log(sub_network.edge[q, i])
            uninfected_likelihood += m_sum
            # for  in range(cascade.node_num):
    # likelihood = infected_likelihood + uninfected_likelihood     # 这一样
    likelihood = infected_likelihood            # 这里不一样
    return likelihood

# 这个函数是求哪个似然？Q(S)? 在文档中没看到对应的
def net_cascade_likelihood_5a_lxm(cascade_id, cascade, sub_network):
    # this is the likelihood of a cascade to a network
    # using 5A
    node_num = cascade.node_num
    cas_label = cascade.labels[cascade_id]
    same_label_cascade, same_label_num = cascade.prune_with_label(cas_label)
    mi_matrix = nt.cal_mi_from_cascade(same_label_cascade, same_label_num, node_num)
    # weight = np.zeros([node_num, node_num])

    sub_sum_mi = np.sum(sub_network.graph * mi_matrix, axis=0)
    sub_sum_mi[sub_sum_mi == 0] = np.inf

    weight = mi_matrix / sub_sum_mi

    q = 1 - sub_network.edge

    q = q * cascade.cascade[cascade_id].reshape(-1, 1)
    q[q == 0] = 1
    sub = 1 - np.prod(q)        # 这点是不是少写了个axis = 0？
    sub = sub.flatten()

    sub[sub == 0] = -1
    f = sub_network.edge / sub
    f[f < 0] = 0

    infected_likelihood = 0
    uninfected_likelihood = 0

    for i in range(cascade.node_num):
        if cascade.cascade[cascade_id, i] == 1:
            m_sum = 0

            for t in range(cascade.node_num):
                if sub_network.edge[t, i] > 0:
                    if cascade.cascade[cascade_id, t] > 0:
                        # prod *= (1 - sub_network.edge[t, i])
                        m_sum += weight[t, i] * (1 - f[t, i]) * np.log(1 - sub_network.edge[t, i])
                        m_sum += weight[t, i] * f[t, i] * np.log(sub_network.edge[t, i])
                    else:
                        m_sum += weight[t, i] * np.log(1 - sub_network.edge[t, i])
            # print("prod = %f " % prod)
            infected_likelihood += m_sum
            # if prod >= 1:
            #     infected_likelihood += m_sum
            # else:
            #     infected_likelihood += (math.log(1 - prod) + m_sum)
        else:
            m_sum = 0
            for q in range(cascade.node_num):
                if sub_network.edge[q, i] > 0 and cascade.cascade[cascade_id, q] > 0:
                    m_sum += weight[q, i] * np.log(1 - sub_network.edge[q, i])
            uninfected_likelihood += m_sum
            # for  in range(cascade.node_num):
    likelihood = infected_likelihood + uninfected_likelihood
    # likelihood = infected_likelihood
    return likelihood


def net_cascade_error_count_score(cascade_id, cascade, sub_network):
    # 感染源节点为不确定因素，故error cnt的方法不能准确判断
    node_num = cascade.node_num
    cas_label = cascade.labels[cascade_id]
    single_cas = cascade.cascade[cascade_id]
    err_cnt = 0
    for i in range(node_num):
        if single_cas[i] == 1:
            if np.sum(sub_network.graph[:, i].flatten() * single_cas) <= 0:
                err_cnt += 1

    return - err_cnt


def net_cascade_failure_score_1(cascade_id, cascade, sub_network):
    # 感染源节点为不确定因素，故error cnt的方法不能准确判断
    # 另外加入 F++ 的感染未成功概率打分
    # 越大符合程度越好
    node_num = cascade.node_num
    single_cas = cascade.cascade[cascade_id]
    failure_rate = 0
    log_q = np.log(1 - sub_network.edge)
    for i in range(node_num):
        if single_cas[i] <= 0:
            continue
        if np.sum(sub_network.graph[:, i].flatten() * single_cas) <= 0:
            failure_rate += 0
        else:
            failure_rate += np.sum(log_q[:, i].flatten() * single_cas)
    return - failure_rate


def net_cascade_failure_score_2(cascade_id, cascade, sub_network, para_active_parent_only=False):
    # 在net_cascade_failure_score_1的基础上将p归一化
    node_num = cascade.node_num
    single_cas = cascade.cascade[cascade_id]
    failure_rate = 0
    if para_active_parent_only:
        arr_sum = np.sum(sub_network.edge * single_cas.reshape([node_num, 1]), axis=0)
    else:
        arr_sum = np.sum(sub_network.edge, axis=0)
    arr_sum[arr_sum <= 0] = np.inf
    arr_sum = sub_network.edge / arr_sum
    arr_sum[arr_sum >= 0.9] = 0.9
    # print(arr_sum)
    # os.system("pause")
    log_q = np.log(1 - arr_sum)
    for i in range(node_num):
        if single_cas[i] <= 0:
            continue
        if np.sum(sub_network.graph[:, i].flatten() * single_cas) <= 0:
            failure_rate += 0
        else:
            failure_rate += np.sum(log_q[:, i].flatten() * single_cas)
    return - failure_rate


def net_cascade_failure_score_3(cascade_id, cascade, sub_network, para_active_parent_only=False, choice=0):
    # 在net_cascade_failure_score_1的基础上将p添加一个权重系数
    node_num = cascade.node_num
    single_cas = cascade.cascade[cascade_id]
    failure_rate = 0

    if choice == 0 or choice == 4 or choice == 5:
        mi_matrix = cascade.mi_matrix[cascade.labels[cascade_id]] * sub_network.graph
        if para_active_parent_only:
            sum_mi = np.sum(mi_matrix * single_cas.reshape([node_num, 1]), axis=0)
        else:
            sum_mi = np.sum(mi_matrix, axis=0)

        sum_mi[sum_mi == 0] = np.inf
        sum_mi = mi_matrix / sum_mi
        sum_mi[np.isnan(sum_mi)] = 0

    epsilon = 1e-16
    # choice: 0: 1-MI/sigmaMI*p   1: 1-p    2: 1-p/sigmap   3: 1-p/sigmap*p   4: 1-MI/sigmaMI    5: (1-p)*MI/sigmaMI
    if choice == 0:
        log_q = np.log(1 - sum_mi * sub_network.edge +epsilon)
    elif choice == 1:
        log_q = np.log(1 - sub_network.edge+epsilon)
    elif choice == 2:
        sigmap = np.sum(sub_network.edge, axis=0)
        sigmap[np.where(sigmap == 0)] = np.inf
        log_q = np.log(1 - sub_network.edge/sigmap + epsilon)
    elif choice == 3:
        sigmap = np.sum(sub_network.edge, axis=0)
        sigmap[np.where(sigmap == 0)] = np.inf
        log_q = np.log(1 - sub_network.edge/sigmap*sub_network.edge + epsilon)
    elif choice == 4:
        log_q = np.log(1 - sum_mi + epsilon)
    elif choice == 5:
        tmp = (1-sub_network.edge) * sum_mi
        tmp[np.where(tmp<0)]=0
        log_q = np.log(tmp + epsilon)


    for i in range(node_num):
        if single_cas[i] <= 0:
            continue
        if np.sum(sub_network.graph[:, i].flatten() * single_cas) <= 0:
            failure_rate += 0
        else:
            failure_rate += np.sum(log_q[:, i].flatten() * single_cas)
    return - failure_rate


def net_cascade_failure_score_4(cascade_id, cascade, sub_network, para_active_parent_only=False):
    # 在net_cascade_failure_score_1的基础上将log 1-p 添加一个权重系数, 而不是直接在p上添加
    node_num = cascade.node_num
    single_cas = cascade.cascade[cascade_id]
    failure_rate = 0
    mi_matrix = cascade.mi_matrix[cascade.labels[cascade_id]] * sub_network.graph
    if para_active_parent_only:
        sum_mi = np.sum(mi_matrix * single_cas.reshape([node_num, 1]), axis=0)
    else:
        sum_mi = np.sum(mi_matrix, axis=0)
    # print(sum_mi)
    sum_mi[sum_mi == 0] = np.inf
    sum_mi = mi_matrix / sum_mi
    # print(sum_mi)
    # os.system("pause")
    # sum_mi[sum_mi <= 0.1] = 0.1
    sum_mi[np.isnan(sum_mi)] = 0

    log_q = sum_mi * np.log((1 - sub_network.edge))
    for i in range(node_num):
        if single_cas[i] <= 0:
            continue
        if np.sum(sub_network.graph[:, i].flatten() * single_cas) <= 0:
            failure_rate += 0
        else:
            failure_rate += np.sum(log_q[:, i].flatten() * single_cas)
    return - failure_rate


"""
TWIND求边
"""
def net_parent_set_reconstruct_aaai(dest_net, src_cascade, dest_label, disp=False):
    dest_net.clear_graph()
    node_num = src_cascade.node_num
    cascade, cas_num = src_cascade.prune_with_label(dest_label)
    # start run the reconstruction
    mi_matrix = nt.cal_mi_from_cascade(cascade, cas_num, node_num)
    mi_matrix[np.where(mi_matrix<0)] = 0
    
    upper_bound = np.ceil(math.log((node_num+1) * math.log(math.e * (node_num+1)/2, 2),2))

    for i in range(node_num):
        if disp:
            print("working on node %d " % i)

        # do k-means for the pruning of candidates
        mi_for_i = mi_matrix[i, :].reshape(-1, 1)  # 第i个点
        mi_for_i[i][0] = np.min(mi_for_i)
        # print("mi = ")
        # print(mi_for_i, end='\n')
        y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(mi_for_i)
        if sum(y_pred) > node_num/2:
            parent_id_set = np.argwhere(y_pred == 0).flatten()
        else:
            parent_id_set = np.argwhere(y_pred == 1).flatten()

        # print("\tfinish Kmeans")

        parent_num = len(parent_id_set)
        # print("KMeans后父节点的个数: %d" % (parent_num))
        
        # parent_set_num = int(math.pow(2, parent_num) - 1)
        # print("\tparent set : ", end='')
        # print(parent_id_set, end='')
        # print("; parent number is: %d, combo number is: %d" % (parent_num, parent_set_num))
        # os.system("pause")

        # upper_bound = cal_upper_bound(cascade, node_num, cas_num, parent_id_set)
        # upper_bound = 4
        # print("\tupper bound = %f " % upper_bound)
        
        if disp:
            print("\tupper bound = %f " % upper_bound)
        # os.system("pause")

        if upper_bound < parent_num:
            parent_id_set = sorted(range(len(mi_for_i)), key=lambda t: mi_for_i[t])[0 - int(upper_bound):]
            parent_num = len(parent_id_set)
            parent_set_num = int(math.pow(2, parent_num) - 1)  # 父节点的组合
            if disp:
                print("\tparent set : ", end='')
                print(parent_id_set, end='')
                print("; parent number is: %d, combo number is: %d" % (parent_num, parent_set_num))
        elif upper_bound >= parent_num:
            # upper bound is greater than parent number
            # add all potential parent into the set
            for par in range(parent_num):
                dest_net.add_edge(parent_id_set[par], i)   # parent_id_set : 第i个节点的父节点集合
            if disp:
                print("\tparent set : ", end='')
                print(parent_id_set)
            continue

        parent_set_matrix = gen_parent_set(parent_id_set, parent_num)  # 获取父节点的组合
        # print(parent_set_matrix)
        current_edge_num = 0

        while True:
            max_score = - np.inf
            max_id = -1
            for potential_parent_set_id in range(parent_set_num):  # 取了score最高的parent_set
                score = g_parent_score(cascade, i, parent_set_matrix[potential_parent_set_id], node_num)
                # print("parent:", end='')
                # print(parent_set_matrix[potential_parent_set_id])
                # print("score = %f" % score)
                if score > max_score:
                    max_score = score
                    max_id = potential_parent_set_id
            if max_id >= 0:

                # print("adding parent set: ", end='')
                # print(parent_set_matrix[max_id])
                # os.system("pause")

                for t in range(len(parent_set_matrix[max_id])):
                    if dest_net.get_edge(parent_set_matrix[max_id][t], i) > 0:
                        continue
                    else:
                        dest_net.add_edge(parent_set_matrix[max_id][t], i)
                        current_edge_num += 1
                parent_set_matrix.pop(max_id)
                parent_set_num -= 1
                # print("edge num = ", end='')
                # print(current_edge_num)
                if current_edge_num >= upper_bound:
                    break
            else:
                break

    return dest_net

"""
TWIND求边
"""
def aaai_construct(dest_net, src_cascade, dest_label, disp=False):
    dest_net.clear_graph()
    node_num = src_cascade.node_num
    cascade, cas_num = src_cascade.prune_with_label(dest_label)
    
    beta = cas_num
    # beta, nodes_num = cascade.shape

    # 计算互信息并切图
    prune_network, MI, tau = MI_prune(cascade)
    print("candidate parents_num = ",np.sum(prune_network, axis=0))

    # 计算上界
    bound = math.log((beta+1) * math.log(math.e * (beta+1)/2, 2),2)
    print("bound = ", bound)

    # 计算候选父节点各种组合的得分，之后greedy地选父节点
    constructed_network = np.zeros(prune_network.shape)
    for i in range(node_num):
        candidate_parents = np.where(prune_network[:,i]==1)[0]
        candidate_size = candidate_parents.size

        if candidate_size<=bound:
            constructed_network[candidate_parents, i] = 1
        else:
            par_comb_sets = []
            par_comb_sets.append((np.array([]), cal_score(i, np.array([]), cascade)))
            for k in range(1,int(bound+1)):
                # 获得父节点个数为k的所有组合
                k_combs = list(combinations(candidate_parents, k))
                for comb in k_combs:
                    score = cal_score(i, np.array(comb), cascade)
                    par_comb_sets.append((np.array(comb), score))

            sorted_sets = sorted(par_comb_sets, key = lambda comb:comb[1], reverse= True)

            for comb in sorted_sets:
                if np.sum(constructed_network[:,i]>=bound):
                    break
                temp_rel = constructed_network[:,i]
                temp_rel[comb[0].astype(int)] = 1
                if np.sum(temp_rel)>bound:
                    continue
                constructed_network[:,i] = temp_rel

    return constructed_network, MI, tau

def MI_prune(record_states):
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[int(record_states[result_index, j]), int(record_states[result_index, k])] += 1

            epsilon = 1e-5
            M00 = state_mat[0, 0] / results_num * math.log(
                state_mat[0, 0] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M01 = state_mat[0, 1] / results_num * math.log(
                state_mat[0, 1] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)
            M10 = state_mat[1, 0] / results_num * math.log(
                state_mat[1, 0] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M11 = state_mat[1, 1] / results_num * math.log(
                state_mat[1, 1] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)

            # MI[j, k] = M00 + M11 - abs(M10) - abs(M01)
            MI[j, k] = M00 + M11 + M10 + M01
            MI[k, j] = MI[j, k]

    # Kmeans 聚类
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1

    return prune_network, MI, tau

def cal_score(child, parents, record_states):
    j_num = pow(2, parents.size)
    count_states = np.zeros((2, j_num))
    records_num = record_states.shape[0]

    for record_index in range(records_num):
        i_state = record_states[record_index, child]
        if j_num>1:
            j_state = record_states[record_index, parents]
        else:
            j_state = np.zeros(1)
        count_states[int(i_state), numpy2dec(j_state)] += 1

    factorial_1 = count_states[0, :]
    factorial_2 = count_states[1, :]
    factorial_sum = np.sum(count_states, axis=0)+1

    log_1 = np.zeros(factorial_1.shape)
    log_2 = np.zeros(factorial_2.shape)
    log_sum = np.zeros(factorial_sum.shape)
    for i in range(j_num):
        cur_log = 0
        for k in range(int(factorial_1[i])):
            cur_log+= math.log(k+1,2)
        log_1[i] = cur_log

        cur_log = 0
        for k in range(int(factorial_2[i])):
            cur_log+= math.log(k+1,2)
        log_2[i] = cur_log

        cur_log = 0
        for k in range(int(factorial_sum[i])):
            cur_log += math.log(k + 1, 2)
        log_sum[i] = cur_log


    temp_result = log_1+log_2-log_sum
    score = np.sum(temp_result)

    return score

def numpy2dec(line):
    j = 0
    for m in range(line.size):
        j = j + pow(2, line.size - 1 - m) * line[m]

    return int(j)



"""
求theta
"""
def get_theta(dest_dict, comb_cascade, cat, sub_network_structure, display=0):
    cascade, cas_num = comb_cascade.prune_with_label(cat)
    if cas_num == 0:
        return dest_dict
    dest_dict.clear()
    node_num = comb_cascade.node_num
    cascade, cas_num = comb_cascade.prune_with_label(cat)
    # fp = open("num.txt", "a+", encoding="utf-8")
    
    for i in range(node_num):
        parent_id_set = np.argwhere(sub_network_structure.graph[:, i] == 1).flatten()  # 父节点集合
        parent_num = len(parent_id_set)
        # print("节点%d有%d个父节点" %(i, parent_num))
        parent_state_num = int(math.pow(2, parent_num))   # 父节点的状态组合
        cmp_parent_with_child = np.array(np.append(parent_id_set, i))
        cmp_full_cas = cascade[:, cmp_parent_with_child.astype(int)]
        parent_and_child_states = np.zeros([node_num])
        
        prior_i_1 = len(np.argwhere(cascade[:, i] == 1).flatten())    # i=1的条数
        p_i_1 = prior_i_1 / cas_num
        # prior_i_0 = len(np.argwhere(cascade[:, i] == 0).flatten()) / cas_num
        j_pattern = np.append(np.zeros(len(parent_id_set)), 1).astype(int)  # 0...01
        prior_j_0 = np.sum(cmp_full_cas == j_pattern, axis=1)       
        kind_j = collections.Counter(prior_j_0)
        prior_i_j_1 = kind_j[len(j_pattern)]     # 父节点状态都为0且i=1的条数
        prior_i_j = (prior_i_1-prior_i_j_1) / cas_num    
        
        # 计数
        # condition1 = 0
        # condition2 = 0
        # condition3 = 0
        # condition4 = 0
        # fp.write("\n")
        
        for j in range(parent_state_num):
            state_str = bin(j)
            state_str = state_str.lstrip('0b')
            parent_and_child_states *= 0
            for k in range(len(state_str)):   # 具体状态值
                if state_str[len(state_str) - k - 1] == '1':
                    parent_and_child_states[parent_id_set[parent_num - k - 1]] = 1
            
            parent_and_child_states[i] = 0
            cmp_dest_pattern = parent_and_child_states[cmp_parent_with_child.astype(int)].flatten()
            res = np.sum(cmp_full_cas == cmp_dest_pattern, axis=1)
            cnter = collections.Counter(res)
            cnt0 = cnter[len(cmp_dest_pattern)] 
            
            parent_and_child_states[i] = 1
            cmp_dest_pattern = parent_and_child_states[cmp_parent_with_child.astype(int)].flatten()
            res = np.sum(cmp_full_cas == cmp_dest_pattern, axis=1)
            cnter = collections.Counter(res)
            cnt1 = cnter[len(cmp_dest_pattern)]
            
            # 统计j中1的个数
            k = sum(cmp_dest_pattern[:parent_num])
            
            key = ''.join(str(int(s)) for s in cmp_dest_pattern[:parent_num])
            
            # if k > 0:
            #     value = 1 - (1-prior_i_j)**k
            # else:
            #     value = 0.1
                
            # if cnt1 > 0 and cnt0 > 0:
            #     value = (cnt1) / (cnt0+cnt1)
            # elif cnt1 == 0 and cnt0 > 0:
            #     value = p_i_1 / (cnt0+cnt1)
            # elif cnt1 == 0 and cnt0 == 0:
            #     value = p_i_1
            # else:
            #     if k > 0:
            #         value = 1 - (1-prior_i_j)**k
            #     else:
            #         value = p_i_1
                
            # value 的取值
            # b = 0.2
            if cnt1==0 and cnt0 > 0: # 这种父节点组合无法造成节点i激活
                # value = (b*prior_i_1) / (cnt0+cnt1+b)   # 取1-value
                # condition2 += 1
                # value = p_i_1 / (cnt0+cnt1)
                value = p_i_1
            elif cnt1 > 0 and cnt0==0:  # 这种父节点组合大概率会造成节点i激活
                # print("-----------")
                # print(cnt1)
                # print("-----------")
                # if cnt1 > 10:
                #     ceo = 0.6
                # else:
                #     ceo = 0.4
                # value = (cnt0+cnt1+b*prior_i_1) / (cnt0+cnt1+b)   # 取value
                # value = b*((cnt0+cnt1)+b*p_i_1) / (cnt0+cnt1)
                # value = p_i_1
                # condition3 += 1
                # value = p_i_1 + ((b*b*p_i_1)/(cnt0+cnt1))
                value = p_i_1 + (0.01 * p_i_1)
            elif cnt1==0 and cnt0==0:   # 父节点组合未出现，不知道是否存在关系
                # condition1 += 1
                value = p_i_1
                # numpy.random.normal(loc=0.0, scale=1.0, size=None)
            else:    # 以一定的概率感染节点i
                # condition4 += 1
                value = (cnt1) / (cnt0+cnt1)
            
            theta = {key : value}
            if j == 0:                # 避免覆盖
                dest_dict.update({i:theta})
            else:
                dest_dict[i].update(theta)
                
        # one_row = np.array([condition1, condition2, condition3, condition4])
        # str_one_row = '\t'.join(str(i) for i in one_row)
        # fp.write(str_one_row)
        # fp.write("\n")
            
        # list_key = np.array(list(dest_dict[i].keys()))
        # list_value = np.array(list(dest_dict[i].values()))  
        # # 父节点组合未出现时，value如何取值
        # sum_value = list_value.sum()
        # zeros = np.argwhere(list_value == 0).flatten()   # 值为0的索引
        # list_value[zeros] = sum_value / (parent_state_num-len(zeros))  # 取平均
        # dest_dict[i] = dict(zip(list_key,list_value))
                
    # fp.close()
    return dest_dict


def gen_parent_set(parent_id_set, parent_num):
    parent_set_matrix = []
    for i in range(1, pow(2, parent_num)):
        bin_id = bin(i)  # 返回二进制表示
        bin_id = bin_id.lstrip('0b')
        tmp_arr = []
        # print("i = %d with \" %s" %( i, bin_id))
        # print(parent_id_set)
        for j in range(len(bin_id)):
            # 倒序查看是否为1
            if bin_id[len(bin_id) - j - 1] == '1':
                tmp_arr.append(parent_id_set[parent_num - j - 1])
        # print(tmp_arr)
        # os.system("pause")
        parent_set_matrix.append(tmp_arr)
    return parent_set_matrix


# 评价标准-F值函数
def EVA_f_score(m_test_network, ground_truth_network, display=1):
    correct_predict = 0
    wrong_predict = 0
    missing = 0
    for i in range(m_test_network.node_num):
        for j in range(m_test_network.node_num):
            if m_test_network.graph[i, j] > 0:
                if ground_truth_network.graph[i, j] > 0:
                    correct_predict += 1
                else:
                    wrong_predict += 1
            elif ground_truth_network.graph[i, j] > 0:
                missing += 1
    precision = correct_predict / (correct_predict + wrong_predict)
    recall = correct_predict / (correct_predict + missing)
    f_score = 2 * precision * recall / (precision + recall)
    if display == 1:
        # print("network f-score result:")
        print("f-score of network with label = %d is %.4f" % (m_test_network.category, f_score))
        print("precision = %f, recall = %f" % (precision, recall))
    return f_score


# 评价标准-MSE函数
def EVA_mse_value(m_test_network, ground_truth_network, display=1):
    mse = 0
    edge_cnt = 0
    for i in range(ground_truth_network.node_num):
        for j in range(ground_truth_network.node_num):
            mse += np.power(ground_truth_network.edge[i, j] - m_test_network.edge[i, j], 2)
            # if ground_truth_network.edge[i, j] > 0:
            #     print("ground truth: %f, inferred: %f" %(ground_truth_network.edge[i, j], m_test_network.edge[i, j]))
            # if ground_truth_network.graph[i, j] > 0:
            #     mse += np.power(ground_truth_network.edge[i, j] - m_test_network.edge[i, j], 2)
            #     edge_cnt += 1
    mse = mse / math.pow(ground_truth_network.node_num, 2)
    if display == 1:
        # print("network MSE result:")
        print("MSE of network with label = %d is %.10f" % (m_test_network.category, mse))
    return mse


# 评价标准-label准确率函数
def EVA_label_accuracy(test_cascade, ground_truth_label_array, display=1):
    assert (len(ground_truth_label_array) == test_cascade.cascade_num)
    tmp_arr = np.fabs(test_cascade.labels - ground_truth_label_array)
    tmp_arr[tmp_arr > 0] = 1
    error = sum(tmp_arr)
    correct = test_cascade.cascade_num - error
    if display == 1:
        print("label accuracy result: ")
        print("right number: %d, wrong number %d, accuracy : %.2f%% " % (
        correct, error, 100 * correct / test_cascade.cascade_num))


# 测试基本成功: 100% correct rate
def g_edge_score_func_test():
    node_num = 200
    cascade_num = 300  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"
    network_name_2with3 = "graph_network_combined_200_2with3.txt"  # for parent relationships
    cascade_name = "record_states_network_200_2with3.txt"
    net2with3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2with3)
    cas2with3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name, cat_num=2)
    net2with3.read_network()
    cas2with3.read_cascade()
    # base line networks
    network_name_2 = "graph_network_200_2.txt"
    network_name_3 = "graph_network_200_3.txt"
    net2 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2)
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net2.read_network()
    net3.read_network()

    # start evaluation
    labels = np.ones([cascade_num])
    for i in range(int(cascade_num / 2)):
        labels[i] = 0
    cas2with3.label_renew(labels)
    correct = error = 0
    min_score = np.inf
    for i in range(node_num):
        for j in range(node_num):
            if net2with3.get_edge(i, j) > 0:
                score_label_0 = g_edge_score(i, j, cas2with3, 0)
                score_label_1 = g_edge_score(i, j, cas2with3, 1)
                if score_label_1 > score_label_0 and net3.get_edge(i, j) >= 0:
                    correct += 1
                    if score_label_1 < min_score:
                        min_score = score_label_1
                elif score_label_0 > score_label_1 and net2.get_edge(i, j) >= 0:
                    correct += 1
                    if score_label_0 < min_score:
                        min_score = score_label_0
                else:
                    error += 1
                label = score_label_0 < score_label_1
                # print("s0 = " + str(score_label_0) + " , s1 = " + str(score_label_1) + ", label = " + str(label))
    print("minimum score: %f" % min_score)
    print("correct number: %d; error number = %d; accuracy: %.2f%%" % (correct, error, 100 * correct / (correct + error)))


# 测试5A网络重构代码正确性
def net_diffusion_rate_5a_func_test():
    node_num = 200
    cascade_num = 150  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"
    network_name_3 = "graph_network_200_3.txt"
    cascade_name_3 = "record_states_network_200_3.txt"
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net3.read_network()
    cas3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name_3, cat_num=1)
    cas3.read_cascade()
    new_net = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    new_net.overwrite_graph(net3.graph)
    net_diffusion_rate_5a_m(cas3, 0, new_net, display=1)
    new_net.display_edges()
    EVA_f_score(new_net, net3)
    EVA_mse_value(new_net, net3)


# 测试aaai网络重构代码正确性
def net_parent_set_reconstruct_aaai_func_test():
    node_num = 200
    cascade_num = 150  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"
    network_name_3 = "graph_network_200_3.txt"
    cascade_name_3 = "record_states_network_200_3.txt"
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net3.read_network()
    cas3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name_3, cat_num=1)
    cas3.read_cascade()
    new_net = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    start = time.clock()
    new_net = net_parent_set_reconstruct_aaai(new_net, cas3, 0)
    end = time.clock()
    EVA_f_score(new_net, net3)
    print("runtime = %f" % (end - start))


# 测试5A网络重构代码正确性
def net_parent_set_reconstruct_5A_func_test():
    node_num = 200
    cascade_num = 150  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"
    network_name_3 = "graph_network_200_3.txt"
    cascade_name_3 = "record_states_network_200_3.txt"
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net3.read_network()
    cas3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name_3, cat_num=1)
    cas3.read_cascade()
    new_net = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    start = time.clock()
    new_net = net_parent_set_reconstruct_5a(new_net, cas3, 0)
    end = time.clock()
    EVA_f_score(new_net, net3)
    print("runtime = %f" % (end - start))


# 测试所有的cascade assignment方法在理想环境下是否能够保持正确
def net_cascade_assignment_func_test():
    node_num = 200
    cascade_num = 300  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"
    network_name_2with3 = "graph_network_combined_200_2with3.txt"  # for parent relationships
    cascade_name = "record_states_network_200_2with3.txt"
    net2with3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr,
                             network_name=network_name_2with3)
    cas2with3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name,
                             cat_num=2)

    network_name_2 = "graph_network_200_2.txt"
    network_name_3 = "graph_network_200_3.txt"
    net2 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2)
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net2.read_network()
    net3.read_network()

    net2with3.read_network()
    cas2with3.read_cascade()
    ground_truth_labels = np.ones([cascade_num])
    for i in range(int(cascade_num / 2)):
        ground_truth_labels[i] = 0
    cas2with3.init_ground_truth_labels(ground_truth_labels)
    cas2with3.label_renew(ground_truth_labels)

    for i in range(cascade_num):
        error_score_0 = net_cascade_failure_score_1(i, cas2with3, net2)
        error_score_1 = net_cascade_failure_score_1(i, cas2with3, net3)
        if error_score_1 > error_score_0:
            choose_ent = 0
        else:
            choose_ent = 1
        print("# %d == for net 0: %.1f; for net 1: %.1f; choose %d" % (i, error_score_0, error_score_1, choose_ent))


# 测试增量式计算MI算法正确
def incremental_mi_correct_test():
    node_num = 200
    cascade_num = 150  # all cascades total
    network_base_addr = "./data/network/"
    record_base_addr = "./data/record/"

    network_name_3 = "graph_network_200_3.txt"
    cascade_name_3 = "record_states_network_200_3.txt"
    net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net3.read_network()
    cas3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name_3, cat_num=1)
    cas3.read_cascade()

    network_name_2 = "graph_network_200_2.txt"
    cascade_name_2 = "record_states_network_200_2.txt"
    net2 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    net2.read_network()
    cas2 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name_3, cat_num=1)
    cas2.read_cascade()

    first_batch_cas_mat = cas3.cascade[0: 120, :]
    second_batch_cas_mat = cas2.cascade[120: 150, :]
    full_cas_mat = np.concatenate((first_batch_cas_mat, second_batch_cas_mat), axis=0)

    whole_mi = nt.cal_mi_from_cascade_full(full_cas_mat, cascade_num, node_num)
    avg_mi = nt.cal_mi_from_cascade_full(first_batch_cas_mat, 120, node_num) * 0.8 + nt.cal_mi_from_cascade_full(second_batch_cas_mat, 30, node_num) * 0.2
   # print(np.argsort(whole_mi))
   # print(np.argsort(avg_mi))

    # print(np.sum(np.argsort(whole_mi)!= np.argsort(avg_mi)))
    # print(np.sum(np.abs(avg_mi - whole_mi)))
    print(np.min(whole_mi)," ", np.max(whole_mi))
    print(np.min(avg_mi), " ", np.max(avg_mi))

    print(np.max(np.abs(whole_mi-avg_mi)))

if __name__ == "__main__":
    
     incremental_mi_correct_test()
    # net_parent_set_reconstruct_5A_func_test()
    # net_parent_set_reconstruct_aaai_func_test()
    # net_parent_set_reconstruct_5A_func_test()


    # print(gen_parent_set([1, 2, 3], 3))

    # net_cascade_assignment_func_test()

    # g_edge_score_func_test()
    # classifier_1_func_test()
    # net_parent_set_reconstruct_aaai_func_test()



    # node_num = 200
    # cascade_num = 300  # all cascades total
    # network_base_addr = "./data/network/"
    # record_base_addr = "./data/record/"
    # network_name_2with3 = "graph_network_combined_200_2with3.txt"  # for parent relationships
    # cascade_name = "record_states_network_200_2with3.txt"
    # net2with3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2with3)
    # cas2with3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name, cat_num=2)
    # net2with3.read_network()
    # cas2with3.read_cascade()

    # network_name_2 = "graph_network_200_2.txt"
    # network_name_3 = "graph_network_200_3.txt"
    # net2 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2)
    # net3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_3)
    # net2.read_network()
    # net3.read_network()


    # # test cascade of edge overlap
    # all_cascade = cas1.get_cascade()
    # one_shot = all_cascade[0,:]
    # infected_idx = np.argwhere(one_shot>=1).flatten()
    # for i in range(len(infected_idx)):
    #     j = i + 1
    #     while j < len(infected_idx):
    #         if net2.get_edge(infected_idx[i], infected_idx[j]) or net2.get_edge(infected_idx[j], infected_idx[i]):
    #             print("hit edge ( " + str(infected_idx[i]) + "," + str(infected_idx[j]) + ") in net2")
    #         if net3.get_edge(infected_idx[i], infected_idx[j]) or net3.get_edge(infected_idx[j], infected_idx[i]):
    #             print("hit edge ( " + str(infected_idx[i]) + "," + str(infected_idx[j]) + ") in net3")
    #         j += 1
    # km_cluster = KMeans(n_clusters=2, max_iter=300, n_init=40, init='k-means++',n_jobs=-1)
