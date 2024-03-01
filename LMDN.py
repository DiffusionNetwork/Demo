# encoding:utf-8
import LMDN_net_func as net_util
import LMDN_network as nt
import numpy as np
import math
import time
import os
from sklearn.cluster import KMeans
from collections import Counter
import itertools as it

'''
graphPathList: aspects的路径
base_addr:
cascade_name: multi-aspects的路径
numberNodes: =beta?
numberAspects: aspects的个数 (cat_num)
cascade_num: multi-aspects 的结点数 = numberAspects * beta
beta: 每个aspect的结点数
'''
def load_data_K(graphPathList, base_addr, cascade_name, numberNodes, numberAspects, cascade_num, beta):
    groundTGraphs = []
    for k in range(numberAspects):
        with open(graphPathList[k],'r') as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            data = np.array([[int(node) for node in line] for line in lines])
            groundTruthGraph = np.zeros((numberNodes,numberNodes))
            numberEdges = data.shape[0]
            for i in range(numberEdges):
                groundTruthGraph[data[i,0]-1,data[i,1]-1]=1   # 有边则为1
            groundTGraphs.append(groundTruthGraph)  # numberAspects个图的结构，即边


    cascades = nt.m_cascade(numberNodes, cascade_num, base_addr=base_addr, cascade_name=cascade_name,
                             cat_num=numberAspects)
    cascades.read_cascade()    # 组合图的感染记录
    ground_truth_labels = np.ones([cascade_num])
    # # for i in range(int(cascade_num)):
    # #     ground_truth_labels[i] = int(i / beta)    # 第i条记录原本属于哪一个aspect --> 真实的aspect label
    # ground_truth_labels[:beta[0]] = 0
    # ground_truth_labels[beta[0]:beta[0]+beta[1]] = 1
    # ground_truth_labels[beta[0]+beta[1]:beta[0]+beta[1]+beta[2]] = 2
    true_label_value = 0
    bi_begin_index = 0
    for bi in range(len(beta)):
        bi_end_index = bi_begin_index + beta[bi]
        ground_truth_labels[bi_begin_index:bi_end_index] = true_label_value
        bi_begin_index = bi_end_index
        true_label_value = true_label_value + 1
    cascades.init_ground_truth_labels(ground_truth_labels)
    return groundTGraphs, cascades   # 返回 真实图，组合图感染记录

def cal_F1_average_mulGraph(networks, groundTGraphs, numberAspects):
    # k个子图，分别计算F1，求平均，不合并
    recall_avg=0
    precision_avg=0
    f1_avg = 0
    f1Mat = np.zeros((numberAspects,numberAspects))
    recallMat = np.zeros((numberAspects, numberAspects))
    precisionMat = np.zeros((numberAspects, numberAspects))
    record_f1 = []
    record_recall = []
    record_precision = []


    tmpList = [i for i in range(numberAspects)]
    permus = list(it.permutations(tmpList))
    epsilon = 1e-16
    for k in range(numberAspects):
        inferGraph = networks[k].graph
        for i in range(numberAspects):
            groundTruthGraph = groundTGraphs[i]
            TP = np.sum((inferGraph + groundTruthGraph) == 2)
            FP = np.sum((inferGraph - groundTruthGraph) == 1)
            FN = np.sum((inferGraph - groundTruthGraph) == -1)

            recallMat[k,i] = TP / (TP + FN + epsilon)
            precisionMat[k,i] = TP / (TP + FP + epsilon)
            f1Mat[k,i] = 2 * recallMat[k,i] * precisionMat[k,i] / (recallMat[k,i] + precisionMat[k,i]+ epsilon)

    maxSum = -np.inf
    maxIndex = -1
    for i in range(len(permus)):
        curPer = permus[i]
        sum = 0
        for k in range(numberAspects):
            sum+=f1Mat[k,curPer[k]]
        if sum>maxSum:
            maxSum = sum
            maxIndex = i

    selPer = permus[maxIndex]

    for k in range(numberAspects):
        recall_avg+=recallMat[k,selPer[k]]
        precision_avg+=precisionMat[k,selPer[k]]
        f1_avg+=f1Mat[k,selPer[k]]

        record_recall.append(recallMat[k,selPer[k]])
        record_precision.append(precisionMat[k,selPer[k]])
        record_f1.append(f1Mat[k,selPer[k]])

    recall_avg/=numberAspects
    precision_avg/=numberAspects
    f1_avg/=numberAspects
    trueIndex = list(selPer)

    return recall_avg, precision_avg, f1_avg, trueIndex, record_recall, record_precision, record_f1


def cal_MSE_mulGraph(networks, groundTGraphs, numberAspects, rateList, trueIndex):
    numberNodes = networks[0].node_num
    MSE = 0
    record_MSE = []
    for k in range(numberAspects):
        inferTransRate = networks[k].edge
        index = trueIndex[k]
        groundTruthTransRate = groundTGraphs[index] * rateList[index]
        cur_MSE = np.sum(np.square(inferTransRate - groundTruthTransRate))/(numberNodes * numberNodes)
        MSE += cur_MSE
        record_MSE.append(cur_MSE)

    return MSE/numberAspects, record_MSE


def cal_MAE_mulGraph(networks, groundTGraphs, numberAspects, rateList, trueIndex):
    MAE = 0
    record_MAE = []
    for k in range(numberAspects):
        inferTransRate = networks[k].edge
        index = trueIndex[k]
        groundTruthTransRate = groundTGraphs[index] * rateList[index]
        midInferTransRate = inferTransRate * groundTGraphs[index]
        tmp = groundTruthTransRate.copy()
        nonZeroCnt = np.sum(groundTGraphs[index])
        tmp[np.where(tmp==0)]=1
        cur_MAE = (np.sum(abs(midInferTransRate - groundTruthTransRate)/tmp)/nonZeroCnt)
        MAE += cur_MAE
        record_MAE.append(cur_MAE)

    return MAE/numberAspects, record_MAE



def cal_NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-16
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


def Ours(groundTGraphs, comb_cascade, cat_num, beta, rateList, debug_display=False, choice=0):
    beginTime = time.time()
    cascade_num = comb_cascade.cascade_num
    node_num = comb_cascade.node_num

    comb_cascade.init_cat_labels()
    
    # 初始 label 隔1个调正确
    # start = 0
    # while start < cascade_num:
    #     comb_cascade.labels[start] = int(start/beta)
    #     start += 2
    
    # beta ratio 初始 label 隔1个调正确
    start = 0
    while start < cascade_num:
        if start < beta[0]:
            comb_cascade.labels[start] = 0
        elif beta[0] <= start < beta[0]+beta[1]:
            comb_cascade.labels[start] = 1
        else:
            comb_cascade.labels[start] = 2
        start += 2

    piValue = np.zeros(cat_num)   
    for cat in range(cat_num):
        piValue[cat] = np.sum(comb_cascade.labels == cat) / cascade_num
    print("initial piValue: ",piValue)

    # print("init with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))


    ground_truth_labels = np.ones([cascade_num])
    # # for i in range(int(cascade_num / 2)):
    # #     ground_truth_labels[i] = 0
    #
    # # for i in range(int(cascade_num)):
    # #     ground_truth_labels[i] = int(i/beta)
    #
    # ground_truth_labels[:beta[0]] = 0
    # ground_truth_labels[beta[0]:beta[0]+beta[1]] = 1
    # ground_truth_labels[beta[0]+beta[1]:beta[0]+beta[1]+beta[2]] = 2
    #
    # # comb_cascade.label_renew(ground_truth_labels)
    true_label_value = 0
    bi_begin_index = 0
    for bi in range(len(beta)):
        bi_end_index = bi_begin_index + beta[bi]
        ground_truth_labels[bi_begin_index:bi_end_index] = true_label_value
        bi_begin_index = bi_end_index
        true_label_value = true_label_value + 1

    sub_net_list = []
    for i in range(cat_num):  # 初始化cat_num个m_network
        tmp_net = nt.m_network(node_num)
        sub_net_list.append(tmp_net)
        tmp_net = None
    theta_dict = []    # 记录每个aspect的各节点(i)各父节点组合(j)的theta值
    for i in range(cat_num):
        tmp_dict = {}
        theta_dict.append(tmp_dict)  
        tmp_dict = None
    cnt = 0
    while True:
        ite_begin_time = time.time()
        cnt += 1
        # first step: generate network
        # 1.1 generate network structure
        # clear all sub-graphs
        for i in range(cat_num):  # sub_net_list: 每个aspect的边
            sub_net_list[i].clear_graph()  #清空，都置为0
            sub_net_list[i] = net_util.net_parent_set_reconstruct_5a(sub_net_list[i], comb_cascade, i, 0)
            # sub_net_list[i] = net_util.net_parent_set_reconstruct_aaai(sub_net_list[i], comb_cascade, i, 0)
            # sub_net_list[i] = net_util.aaai_construct(sub_net_list[i], comb_cascade, i, 0)
            print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
        # os.system("pause")

        # 1.2 update theta
        for i in range(cat_num):
            theta_dict[i].clear()
            theta_dict[i] = net_util.get_theta(theta_dict[i], comb_cascade, i, sub_net_list[i], display=0)
            # sub_net_list[cat].display_edges()
            print("finish calculating theta for graph %d" % i)
            # os.system("pause")
        
        # import json
        # jsObj = json.dumps(theta_dict, indent=4)  # indent参数是换行和缩进
        # fileObject = open('1.json', 'w')
        # fileObject.write(jsObj) 
        # fileObject.close()  # 最终写入的json文件格式

        # second step: re-assign labels of cascades
        finish_label = True
        record_score = np.zeros(node_num)
        record_value_i = np.zeros(node_num)
        record_aspect = np.zeros(cat_num)
                    
        
        for c in range(cascade_num):
            max_likelihood = -np.inf
            max_label = -1
            for cat in range(cat_num):
                score = 1
                for i in range(node_num):
                    parent_id_set = np.argwhere(sub_net_list[cat].graph[:, i] == 1).flatten()  # 父节点集合
                    parent_state = comb_cascade.cascade[c][parent_id_set.astype(int)]  # 从这条记录中取出父节点的状态
                    # 取出theta值
                    key = ''.join(str(int(s)) for s in parent_state)
                    # print('c = ' + str(c) + ', cat = ' + str(cat) + ', i = ' + str(i))
                    # if (c == 0) and (cat == 1) and (i == 0):
                    #     print('c = ' + str(c) + ', cat = ' + str(cat) + ', i = ' + str(i))
                    value = theta_dict[cat][i].get(key) 
                    if comb_cascade.cascade[c][i] == 1:
                        value_i = value
                    else:
                        value_i = 1-value
                    score *= value_i
                    record_value_i[i] = value_i
                    record_score[i] = score
                record_aspect[cat] = score
                if score > max_likelihood:
                    max_likelihood = score
                    max_label = cat
            if finish_label and max_label != comb_cascade.labels[c]:
                finish_label = False
            comb_cascade.labels[c] = max_label



        if debug_display:
            ite_end_time = time.time()
            ite_time = ite_end_time - ite_begin_time

            print("iter %d times" % cnt)
            print("curIteTime cost: ",ite_time*1000)
            # print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))

            # F1, MSE, MAE, NMI
            recall, precision, f1, trueIndex, record_recall, record_precision, record_f1 = cal_F1_average_mulGraph(sub_net_list, groundTGraphs, cat_num)
            MSE, record_MSE = cal_MSE_mulGraph(sub_net_list, groundTGraphs, cat_num, rateList, trueIndex)
            MAE, record_MAE = cal_MAE_mulGraph(sub_net_list, groundTGraphs, cat_num, rateList, trueIndex)
            NMI = cal_NMI(comb_cascade.labels, ground_truth_labels)

            print("record_f1: ",record_f1)
            print("f1_std = %.5f"%(np.std(record_f1)))
            print("recall=%.3f, precision=%.3f, f1=%.3f" % (recall, precision, f1))

            print("record_MSE: ",record_MSE)
            print("MSE_std = %.5f"%(np.std(record_MSE)))
            print("MSE=%.5f" % (MSE))

            print("record_MAE: ",record_MAE)
            print("MAE_std = %.5f"%(np.std(record_MAE)))
            print("MAE=%.5f" % (MAE))

            print("NMI=%.5f" % (NMI))

            piValue = np.zeros(cat_num)
            for cat in range(cat_num):
                piValue[cat] = np.sum(comb_cascade.labels==cat)/cascade_num
            print(piValue)

            print(comb_cascade.labels)
            print("\n\n")
        else:
            print("iter %d times" % cnt)

        midTime = time.time()
        print("curTime cost: ",(midTime-beginTime)*1000)

        # if cnt < 15:
        #     finish_label = False

        if finish_label:
            endTime = time.time()
            print("time cost: ",(endTime-beginTime)*1000)
            print("\n\n\nProcess finished! \n\n\n")
            break




# def classifier_exp1(comb_net, comb_cascade, cat_num, debug_display=False, debug_min_score=-np.inf):
#     comb_cascade.init_cat_labels()
#     node_num = comb_net.node_num
#     cascade_num = comb_cascade.cascade_num
#     sub_net_list = []
#     for i in range(cat_num):
#         tmp_net = nt.m_network(node_num)
#         sub_net_list.append(tmp_net)
#         tmp_net = None
#     cnt = 0
#     while True:
#         cnt += 1
#         # first step: generate network
#         # 1.1 generate network structure
#         # clear all sub-graphs
#         for i in range(cat_num):
#             sub_net_list[i].clear_graph()
#         for i in range(node_num):
#             for j in range(node_num):
#                 if comb_net.get_edge(i, j) > 0:
#                     max_score = - np.inf
#                     max_label = -1
#                     for cat in range(cat_num):
#                         current_score = net_util.g_edge_score(i, j, comb_cascade, cat)
#                         # print("%f " % current_score, end='')
#                         if current_score > max_score:
#                             max_label = cat
#                             max_score = current_score
#                     # if max_score < debug_min_score:
#                     #     print("\nedge not assigned")
#                     # else:
#                     sub_net_list[max_label].add_edge(i, j)
#                         # print("\nassign edge to network %d" % max_label)
#         for i in range(cat_num):
#             print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
#         # os.system("pause")
#         # 1.2 generate network edge
#         for cat in range(cat_num):
#             net_util.net_diffusion_rate_5a_m(comb_cascade, cat, sub_net_list[cat], display=0)
#
#         # second step: re-assign labels of cascades
#         finish_label = True
#         for i in range(cascade_num):
#             max_likelihood = - np.inf
#             max_label = -1
#             for cat in range(cat_num):
#                 current_likelihood = net_util.net_cascade_error_count_score(i, comb_cascade, sub_net_list[cat])
#                 # print("%f " % current_likelihood, end='')
#                 if current_likelihood > max_likelihood:
#                     max_likelihood = current_likelihood
#                     max_label = cat
#             # print("choose: %d" % max_label)
#             if finish_label and max_label != comb_cascade.labels[i]:
#                 finish_label = False
#             comb_cascade.labels[i] = max_label
#
#         if debug_display:
#             print("iter %d times" % cnt, end='')
#             print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))
#         else:
#             print("iter %d times" % cnt)
#         if finish_label:
#             print("\n\n\nProcess finished! \n\n\n")
#             break
#     return comb_cascade.cal_labels_accuracy()
#
#
# def classifier_exp2(comb_net, comb_cascade, cat_num, debug_display=False):
#     comb_cascade.init_cat_labels()
#     cascade_num = comb_cascade.cascade_num
#     node_num = comb_net.node_num
#
#     ground_truth_labels = np.ones([cascade_num])
#     for i in range(int(cascade_num / 2)):
#         ground_truth_labels[i] = 0
#     # comb_cascade.label_renew(ground_truth_labels)
#
#     sub_net_list = []
#     for i in range(cat_num):
#         tmp_net = nt.m_network(node_num)
#         sub_net_list.append(tmp_net)
#         tmp_net = None
#     cnt = 0
#     while True:
#         cnt += 1
#         # first step: generate network
#         # 1.1 generate network structure
#         # clear all sub-graphs
#         for i in range(cat_num):
#             sub_net_list[i].clear_graph()
#             sub_net_list[i] = net_util.net_parent_set_reconstruct_5a(sub_net_list[i], comb_cascade, i, 0)
#             print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
#         # os.system("pause")
#
#         # 1.2 generate network edge
#         for cat in range(cat_num):
#             net_util.net_diffusion_rate_5a_m(comb_cascade, cat, sub_net_list[cat], display=0)
#             # sub_net_list[cat].display_edges()
#             print("finish calculating diffusion rates for graph %d" % cat)
#             # os.system("pause")
#
#         # second step: re-assign labels of cascades
#         finish_label = True
#         for i in range(cascade_num):
#             max_likelihood = - np.inf
#             max_label = -1
#             for cat in range(cat_num):
#                 current_likelihood = net_util.net_cascade_failure_score_2(i, comb_cascade, sub_net_list[cat], False)
#                 # print("%f " % current_likelihood, end='')
#                 if current_likelihood > max_likelihood:
#                     max_likelihood = current_likelihood
#                     max_label = cat
#             # print("choose: %d" % max_label)
#             if finish_label and max_label != comb_cascade.labels[i]:
#                 finish_label = False
#             comb_cascade.labels[i] = max_label
#
#         if debug_display:
#             print("iter %d times" % cnt, end='')
#             print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))
#         else:
#             print("iter %d times" % cnt)
#         if finish_label:
#             print("\n\n\nProcess finished! \n\n\n")
#             break
#     return comb_cascade.cal_labels_accuracy()


def classifier_exp3(comb_net, comb_cascade, cat_num, debug_display=False):
    comb_cascade.init_cat_labels()
    print("init with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))
    cascade_num = comb_cascade.cascade_num
    node_num = comb_net.node_num

    ground_truth_labels = np.ones([cascade_num])
    for i in range(int(cascade_num / 2)):
        ground_truth_labels[i] = 0
    # comb_cascade.label_renew(ground_truth_labels)

    sub_net_list = []
    for i in range(cat_num):
        tmp_net = nt.m_network(node_num)
        sub_net_list.append(tmp_net)
        tmp_net = None
    cnt = 0
    while True:
        cnt += 1
        # first step: generate network
        # 1.1 generate network structure
        # clear all sub-graphs
        for i in range(cat_num):
            sub_net_list[i].clear_graph()
            sub_net_list[i] = net_util.net_parent_set_reconstruct_5a(sub_net_list[i], comb_cascade, i, 0)
            print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
        # os.system("pause")

        # 1.2 generate network edge
        for cat in range(cat_num):
            net_util.net_diffusion_rate_5a_m(comb_cascade, cat, sub_net_list[cat], display=0)
            # sub_net_list[cat].display_edges()
            print("finish calculating diffusion rates for graph %d" % cat)
            # os.system("pause")

        # second step: re-assign labels of cascades
        finish_label = True
        comb_cascade.cal_mi_for_all_cas()
        for i in range(cascade_num):
            max_likelihood = - np.inf
            max_label = -1
            for cat in range(cat_num):
                current_likelihood = net_util.net_cascade_failure_score_3(i, comb_cascade, sub_net_list[cat], False)
                # print("%f " % current_likelihood, end='')
                if current_likelihood > max_likelihood:
                    max_likelihood = current_likelihood
                    max_label = cat
            # print("choose: %d" % max_label)
            if finish_label and max_label != comb_cascade.labels[i]:
                finish_label = False
            comb_cascade.labels[i] = max_label

        if debug_display:
            print("iter %d times" % cnt, end='')
            print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))
        else:
            print("iter %d times" % cnt)
        if finish_label:
            print("\n\n\nProcess finished! \n\n\n")
            break
    return comb_cascade.cal_labels_accuracy()


# 直接用互信息的值作为权重
def classifier_exp3_no_rates(comb_net, comb_cascade, cat_num, debug_display=False):
    comb_cascade.init_cat_labels()
    cascade_num = comb_cascade.cascade_num
    node_num = comb_net.node_num

    # ground_truth_labels = np.ones([cascade_num])
    # for i in range(int(cascade_num / 2)):
    #     ground_truth_labels[i] = 0
    # comb_cascade.label_renew(ground_truth_labels)

    sub_net_list = []
    for i in range(cat_num):
        tmp_net = nt.m_network(node_num)
        sub_net_list.append(tmp_net)
    cnt = 0
    while True:
        cnt += 1
        # first step: generate network
        # 1.1 generate network structure
        # clear all sub-graphs
        for i in range(cat_num):
            sub_net_list[i].clear_graph()
            sub_net_list[i] = net_util.net_parent_set_reconstruct_5a(sub_net_list[i], comb_cascade, i, 0)
            print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
        # os.system("pause")

        # # 1.2 generate network edge
        # for cat in range(cat_num):
        #     net_util.net_diffusion_rate_5a_m(comb_cascade, cat, sub_net_list[cat], display=0)
        #     sub_net_list[cat].display_edges()
        #     print("finish calculating diffusion rates for graph %d" % cat)
        #     # os.system("pause")
        for cat in range(cat_num):
            pruned_cas, pruned_cas_num = comb_cascade.prune_with_label(cat)
            mi_matrix = nt.cal_mi_from_cascade(pruned_cas, pruned_cas_num, node_num)
            sub_net_list[cat].overwrite_edge(mi_matrix * sub_net_list[cat].graph)
            print(np.sum(sub_net_list[cat].edge))
            os.system("pause")

        # second step: re-assign labels of cascades
        finish_label = True
        for i in range(cascade_num):
            max_likelihood = - np.inf
            max_label = -1
            for cat in range(cat_num):
                current_likelihood = net_util.net_cascade_failure_score_4(i, comb_cascade, sub_net_list[cat])
                print("%f " % current_likelihood, end='')
                if current_likelihood > max_likelihood:
                    max_likelihood = current_likelihood
                    max_label = cat
            print("choose: %d" % max_label)
            if finish_label and max_label != comb_cascade.labels[i]:
                finish_label = False
            comb_cascade.labels[i] = max_label

        if debug_display:
            print("iter %d times" % cnt, end='')
            print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))
        else:
            print("iter %d times" % cnt)
        if finish_label:
            print("\n\n\nProcess finished! \n\n\n")
            break
    return comb_cascade.cal_labels_accuracy()


# 测试整体classifier的正确性
# def classifier_1_func_test():
#     node_num = 200
#     cascade_num = 300  # all cascades total
#     network_base_addr = "./data/network/"
#     record_base_addr = "./data/record/"
#     network_name_2with3 = "graph_network_combined_200_2with3.txt"  # for parent relationships
#     cascade_name = "record_states_network_200_2with3.txt"
#     net2with3 = nt.m_network(node_num, probability=0.3, base_addr=network_base_addr, network_name=network_name_2with3)
#     cas2with3 = nt.m_cascade(node_num, cascade_num, base_addr=record_base_addr, cascade_name=cascade_name, cat_num=2)
#     net2with3.read_network()
#     cas2with3.read_cascade()
#     ground_truth_labels = np.ones([cascade_num])
#     for i in range(int(cascade_num/2)):
#         ground_truth_labels[i] = 0
#     cas2with3.init_ground_truth_labels(ground_truth_labels)
#     classifier_exp1(net2with3, cas2with3, 2, debug_display=True, debug_min_score=-92)


# 测试整体classifier的正确性
def classifier_2_func_test():
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
    net2with3.read_network()
    cas2with3.read_cascade()
    ground_truth_labels = np.ones([cascade_num])
    for i in range(int(cascade_num / 2)):
        ground_truth_labels[i] = 0
    cas2with3.init_ground_truth_labels(ground_truth_labels)
    return classifier_exp3(net2with3, cas2with3, 2, debug_display=True)


# 测试整体classifier的正确性
def classifier_3_func_test():
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
    net2with3.read_network()
    cas2with3.read_cascade()
    ground_truth_labels = np.ones([cascade_num])
    for i in range(int(cascade_num / 2)):
        ground_truth_labels[i] = 0
    cas2with3.init_ground_truth_labels(ground_truth_labels)
    return classifier_exp3_no_rates(net2with3, cas2with3, 2, debug_display=True)


# acc_sum = 0
# try_time = 3
# for i in range(3):
#     current_acc = classifier_2_func_test()
#     # if current_acc > 1 - current_acc:
#     #     acc_sum += current_acc
#     # acc_sum += max(current_acc, 1 - current_acc)
#     acc_sum += current_acc
# print("----------- \nAverage accuracy is %f " % (100 * acc_sum / try_time))

