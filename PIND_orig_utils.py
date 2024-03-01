from cmath import nan
from re import T
import numpy as np
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import math
import time
import re

def load_data(graph_path, result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[int(node) for node in line] for line in lines])
        ground_truth_network = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[data[i, 0] - 1, data[i, 1] - 1] = 1

    return ground_truth_network, diffusion_result


def generate_prob_result(diffusion_result,mean,scale,prob_result_path,read_flag=False):
    if read_flag:
        prob_result=np.loadtxt(prob_result_path,delimiter=' ')
    elif not read_flag:
        beta,nodes_num=diffusion_result.shape
        bias=np.random.normal(mean,scale,(beta,nodes_num))
        bias[np.where(diffusion_result==1)]*=-1
        prob_result=diffusion_result+bias
        prob_result[np.where(prob_result>1)]=1
        prob_result[np.where(prob_result<0)]=0
        np.savetxt(prob_result_path,prob_result,fmt='%f',delimiter=' ')

    return prob_result


def generate_prob_result_dirichlet(diffusion_result,noise_ratio,prob_result_path,read_flag=False):
    if read_flag:
        prob_result=np.loadtxt(prob_result_path,delimiter=' ')
    elif not read_flag:
        epsilon=1e-8
        beta,nodes_num=diffusion_result.shape
        one_concentration_parameter=[noise_ratio/2+epsilon, 1-noise_ratio/2]
        zero_concentration_parameter=[1-noise_ratio/2, noise_ratio/2+epsilon]
        one_sample=np.random.dirichlet(one_concentration_parameter,(beta,nodes_num))[:,:,1]
        zero_sample=np.random.dirichlet(zero_concentration_parameter,(beta,nodes_num))[:,:,0]
        prob_result=np.zeros(diffusion_result.shape)
        prob_result+=one_sample*diffusion_result
        prob_result+=(1-zero_sample)*(1-diffusion_result)
        np.savetxt(prob_result_path,prob_result,fmt='%f',delimiter=' ')

    return prob_result


def init_p(diffusion_result):
    nodes_num=diffusion_result.shape[1]
    p_matrix = np.random.rand(nodes_num,nodes_num)
    for i in range(nodes_num):
        p_matrix[i,i]=0
    return p_matrix


def prune_init_p(diffusion_result, prune_network):
    nodes_num=diffusion_result.shape[1]
    p_matrix = np.random.rand(nodes_num,nodes_num)*prune_network
    return p_matrix


def myfunc_x(x,*args):
    index_j, p_matrix, prob_result, par_index=args
    par_size=par_index.size
    equation_list=[]
    epsilon=1e-20
    # cal the last item
    p_j_par=p_matrix[par_index,index_j]
    temp_prob_result=prob_result[:,par_index].copy()
    temp_prob_result=x.reshape((1,-1))*np.log(1-temp_prob_result*p_j_par.reshape((1,-1))+epsilon)
    third_item=np.sum(temp_prob_result,axis=1)

    for i in range(par_size):
        cur_par_index=par_index[i]
        prob_i=prob_result[:,cur_par_index]
        prob_j=prob_result[:,index_j]
        pij=p_matrix[cur_par_index,index_j]
        first_item=-np.log(1-prob_i*pij+epsilon)
        second_item=np.log(1-prob_j+epsilon)
        i_equation=np.sum(first_item*(second_item-third_item))
        equation_list.append(i_equation)

    res = np.squeeze(np.array(equation_list))

    return res


def gd_update_edge(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold, prune_network, only_parent):
    # gradient descent
    print("gradient descent updating edge matrix......")
    beta, nodes_num = prob_result.shape
    pre_edge_matrix = edge_matrix.copy()
    epsilon = 1e-20
    max_delta_edge = np.inf
    it_cnt = 0
    while max_delta_edge > stop_threshold:
        temp_gradient = np.zeros((beta, nodes_num, nodes_num))
        for res_index in range(beta):
            # first item
            prob_l = prob_result[res_index, :]
            first_item = -np.log(1 - prob_l.reshape((-1, 1)) * p_matrix+epsilon)

            # second item
            second_item = np.log(1 - np.repeat(prob_l.reshape((1, -1)), nodes_num, axis=0) + epsilon)

            # third item
            temp_third = edge_matrix.copy()
            for i in range(nodes_num):
                temp_third[i,i]=0
            if only_parent:
                temp_third*=prune_network
            temp_third *= np.log(1 - prob_l.reshape((-1, 1)) * p_matrix + epsilon)
            third_item = np.repeat(np.sum(temp_third, axis=0).reshape((1, -1)), nodes_num, axis=0)

            temp_gradient[res_index] = first_item * (second_item - third_item)

        if only_parent:
            gradient_matrix = np.sum(temp_gradient, axis=0)*prune_network
        else:
            gradient_matrix = np.sum(temp_gradient, axis=0)


        edge_matrix -= learning_rate * gradient_matrix
        edge_matrix[np.where(edge_matrix < 0)] = 0
        edge_matrix[np.where(edge_matrix > 1)] = 1
        max_delta_edge = np.max(abs(edge_matrix - pre_edge_matrix))
        sum_delta_edge = np.sum(abs(edge_matrix - pre_edge_matrix))
        pre_edge_matrix = edge_matrix.copy()
        print("%d th iteration, max_delta_edge=%f, sum_delta_edge=%f" % (it_cnt, max_delta_edge, sum_delta_edge))
        it_cnt += 1

        if it_cnt > 50:
            break

    return edge_matrix


def all_update_edge(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold, prune_network):
    new_edge_matrix = np.zeros(p_matrix.shape)
    nodes_num=p_matrix.shape[0]
    gd_update_flag=False
    for j in range(nodes_num):
        print("cal edge for node",j)
        edge_initial = np.zeros(nodes_num)
        prune_par=np.where(prune_network[:,j]==1)[0]
        edge_initial[prune_par]=1
        par_index=np.array([i for i in range(nodes_num)])
        edge_j=fsolve(myfunc_x, edge_initial, (j, p_matrix, prob_result,par_index))
        if not np.sum(edge_j < 0) > 0 and not np.sum(edge_j > 1) > 0:  # no <0 and >1
            print("parents of node %d can be solved from equations" % (j))
            new_edge_matrix[par_index, j] = edge_j

        else:
            # gradient descent
            if not gd_update_flag:
                gd_edge_matrix = gd_update_edge(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold,
                                                prune_network,
                                                only_parent=False)
                gd_update_flag = True
            new_edge_matrix[par_index, j] = gd_edge_matrix[par_index, j].copy()
        new_edge_matrix[j,j]=0

    return new_edge_matrix



def only_parent_update_edge(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold, prune_network):
    new_edge_matrix = np.zeros(p_matrix.shape)
    nodes_num=p_matrix.shape[0]
    gd_update_flag=False
    for j in range(nodes_num):
        # print("cal edge for node",j)
        par_index=np.where(prune_network[:,j]==1)[0]
        par_size=par_index.size
        if par_size<=0:
            continue
        edge_initial = np.ones(par_size)
        edge_j=np.array(fsolve(myfunc_x, edge_initial, (j, p_matrix, prob_result, par_index)))
        if not np.sum(edge_j<0)>0 and not np.sum(edge_j>1)>0:  # no <0 and >1
            print("parents of node %d can be solved from equations"%(j))
            new_edge_matrix[par_index,j]=edge_j
        else:
            # gradient descent
            if not gd_update_flag:
                gd_edge_matrix = gd_update_edge(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold, prune_network,
                           only_parent=True)
                gd_update_flag=True
            new_edge_matrix[par_index,j]=gd_edge_matrix[par_index,j].copy()


    return new_edge_matrix



def update_p_matrix(p_matrix, edge_matrix, prob_result, learning_rate, stop_threshold, prune_network, only_parent):
    # gradient descent
    print("updating p matrix......")
    beta, nodes_num=prob_result.shape
    pre_p_matrix=p_matrix.copy()
    epsilon=1e-20
    max_delta_p=np.inf
    it_cnt=0
    while max_delta_p>stop_threshold:
        temp_gradient=np.zeros((beta,nodes_num,nodes_num))
        for res_index in range(beta):
            # first item
            prob_l=prob_result[res_index,:]
            numerator=edge_matrix*prob_l.reshape((-1,1))
            denominator=1-prob_l.reshape((-1,1))*p_matrix
            first_item=numerator/(denominator+epsilon)

            # second item
            second_item=np.log(1-np.repeat(prob_l.reshape((1,-1)),nodes_num,axis=0)+epsilon)

            # third item
            temp_third = edge_matrix.copy()
            for i in range(nodes_num):
                temp_third[i,i]=0
            if only_parent:
                temp_third *= prune_network

            temp_third=temp_third*np.log(1-prob_l.reshape((-1,1))*p_matrix+epsilon)
            third_item=np.repeat(np.sum(temp_third,axis=0).reshape((1,-1)),nodes_num,axis=0)

            temp_gradient[res_index]=first_item*(second_item-third_item)

        if only_parent:
            gradient_matrix = np.sum(temp_gradient, axis=0)*prune_network
        else:
            gradient_matrix = np.sum(temp_gradient, axis=0)

        p_matrix=p_matrix-learning_rate*gradient_matrix
        p_matrix[np.where(p_matrix < 0)] = 0
        p_matrix[np.where(p_matrix > 1)] = 1
        max_delta_p=np.max(abs(p_matrix-pre_p_matrix))
        sum_delta_P=np.sum(abs(p_matrix-pre_p_matrix))
        pre_p_matrix=p_matrix.copy()
        print("%d th iteration, max_delta_p=%f, sum_delta_p=%f"%(it_cnt,max_delta_p,sum_delta_P))
        it_cnt+=1




        if it_cnt>50:
            break
        # 调试用
        # draw hist distribution
        # print("gradient max=%f,min=%f,mean=%f,mid=%f"%(np.max(gradient_matrix),np.min(gradient_matrix),np.mean(gradient_matrix),np.median(gradient_matrix)))
        # # plt.figure(1000)
        # draw_gradient = gradient_matrix.reshape((1,-1)).tolist()
        # plt.hist(draw_gradient,bins=100)
        #
        # plt.figure(1)
        # draw_p = p_matrix.reshape((1,-1)).tolist()
        # plt.hist(draw_p, bins=100)
        # plt.show()

    return p_matrix


def kmeans_round(prob_edge_matrix):
    # Kmeans 聚类
    nodes_num=prob_edge_matrix.shape[0]
    tmp_edge = prob_edge_matrix.reshape((-1,1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_edge)
    label_pred = estimator.labels_
    temp_0 = tmp_edge[label_pred == 0]
    temp_1 = tmp_edge[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(prob_edge_matrix > tau)] = 1

    return prune_network


def cal_F1(ground_truth_network, inferred_network):
    TP = np.sum(ground_truth_network + inferred_network == 2)
    FP = np.sum(ground_truth_network - inferred_network == -1)
    FN = np.sum(ground_truth_network - inferred_network == 1)
    epsilon = 1e-20
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f_score = 2 * precision * recall / (precision + recall + epsilon)

    return precision, recall, f_score


def cal_mae(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    temp = gt_p.copy()
    temp[temp==0]=1
    temp_infer_p=infer_p.copy()
    temp_infer_p = groundtruth_network*temp_infer_p

    mae = np.sum(abs(temp_infer_p-gt_p)/temp)/edges_num

    return mae


def cal_mae_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    temp_infer_p = groundtruth_network*infer_p

    mae_v2 = np.sum(abs(temp_infer_p-gt_p))/edges_num
    
    return mae_v2


def cal_mse(p, infer_p):
    mse = np.mean(np.square(p-infer_p))

    return mse


def cal_mse_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    mse_v2=np.sum(np.square(groundtruth_network*infer_p-gt_p))/edges_num

    return mse_v2


def modify_p(groundtruth_network, infer_p):
    edges_num=np.sum(groundtruth_network)
    temp=infer_p*groundtruth_network
    mean_p=np.sum(temp)/edges_num
    modified_p=infer_p/mean_p*0.3

    return modified_p
    


# def draw_values(ground_truth_network, value_matrix):
#     temp1=np.squeeze(ground_truth_network.reshape((1,-1)))
#     temp2=np.squeeze(value_matrix.reshape((1,-1)))
#     edge_list=[]
#     for i in range(temp1.size):
#         edge_list.append((temp2[i],temp1[i]))
#     sorted_edge=sorted(edge_list, key=lambda x: x[0])
#     green_index = []
#     green_value = []
#     red_index = []
#     red_value = []
#     for i in range(len(sorted_edge)):
#         if sorted_edge[i][1]==0:
#             red_index.append(i)
#             red_value.append(sorted_edge[i][0])
#         else:
#             green_index.append(i)
#             green_value.append(sorted_edge[i][0])
#
#     plt.scatter(red_index,red_value,color='r')
#     plt.scatter(green_index,green_value,color='g')
#     plt.show()


def weighted_mi_prune(record_states, prune_choice):
    # prune_choice = 0  mi   ;  prune_choice = 1 imi

    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[0,0]+=(1-record_states[result_index,j])*(1-record_states[result_index,k])
                state_mat[0,1]+=(1-record_states[result_index,j])*record_states[result_index,k]
                state_mat[1,0]+=record_states[result_index,j]*(1-record_states[result_index,k])
                state_mat[1,1]+=record_states[result_index,j]*record_states[result_index,k]

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

            if prune_choice==0:
                MI[j, k] = M00 + M11 +M10 + M01
            else:
                MI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            MI[k, j] = MI[j, k]

    # Kmeans 聚类
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    # tmp_MI = tmp_MI[np.where(tmp_MI>0)].reshape((-1,1)) # 在这里切的时候只考虑大于0的值（跟tends的java版本一致）
    #                                                          # 只考虑正数的情况和一刀切，考虑0值的情况在beta=100(120),150的时候有所差别，其余无明显差别

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


    for i in range(nodes_num):
        prune_network[i,i]=0

    # for i in range(nodes_num):
    #     candi_nodes_num=np.sum(prune_network[:,i])
    #     if candi_nodes_num>10:
    #         sorted_index=np.argsort(MI[:,i])
    #         top10_par=sorted_index[-10:]
    #         prune_network[:,i]=0
    #         prune_network[top10_par,i]=1

    return prune_network


def test_weighted_mi_prune(record_states, prune_choice, gt_network):
    # prune_choice = 0  mi   ;  prune_choice = 1 imi

    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[0,0]+=(1-record_states[result_index,j])*(1-record_states[result_index,k])
                state_mat[0,1]+=(1-record_states[result_index,j])*record_states[result_index,k]
                state_mat[1,0]+=record_states[result_index,j]*(1-record_states[result_index,k])
                state_mat[1,1]+=record_states[result_index,j]*record_states[result_index,k]

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

            if prune_choice==0:
                MI[j, k] = M00 + M11 +M10 + M01
            else:
                MI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            MI[k, j] = MI[j, k]

    # Kmeans 聚类
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    # tmp_MI = tmp_MI[np.where(tmp_MI>0)].reshape((-1,1)) # 在这里切的时候只考虑大于0的值（跟tends的java版本一致）
    #                                                          # 只考虑正数的情况和一刀切，考虑0值的情况在beta=100(120),150的时候有所差别，其余无明显差别

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    # draw_values(gt_network, MI)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    # for i in range(nodes_num):
    #     candi_nodes_num=np.sum(prune_network[:,i])
    #     if candi_nodes_num>10:
    #         sorted_index=np.argsort(MI[:,i])
    #         top10_par=sorted_index[-10:]
    #         prune_network[:,i]=0
    #         prune_network[top10_par,i]=1

    return prune_network


def sample_x(x_edge_matrix, sample_times):
    # sampel x_edge_matrix
    x_matrix_list = []
    for i in range(sample_times):
        cur_x=np.zeros(x_edge_matrix.shape)
        sample = np.random.rand(*x_edge_matrix.shape)
        one_index = np.where(sample < x_edge_matrix)
        cur_x[one_index] = 1
        x_matrix_list.append(cur_x.copy())

    return x_matrix_list


def sample_result_v2(prob_result,inner_times, outer_times):
    inner_list=[]
    outer_list=[]

    for i in range(outer_times):
        inner_list.clear()
        sum_sample=np.zeros(prob_result.shape)
        outer_sample=np.zeros(prob_result.shape)
        for j in range(inner_times):
            cur_sample=np.random.rand(*prob_result.shape)
            one_index=np.where(cur_sample < prob_result)
            sum_sample[one_index]+=1
        outer_index=np.where(sum_sample>inner_times/2)
        outer_sample[outer_index]=1
        outer_list.append(outer_sample)

    return outer_list




def cal_loss(x_edge_matrix, p_matrix, prob_result):
    first_item=1-prob_result
    second_item=np.zeros(prob_result.shape)

    epsilon=1e-10
    beta,nodes_num=prob_result.shape
    for record_index in range(beta):
        temp_third = x_edge_matrix.copy()
        prob_l=prob_result[record_index,:].copy()
        for i in range(nodes_num):
            temp_third[i, i] = 0

        temp_third = temp_third * np.log(1 - prob_l.reshape((-1, 1)) * p_matrix + epsilon)
        sum_item = np.sum(temp_third, axis=0).reshape((1, -1))
        second_item[record_index,:]=sum_item.copy()

    loss=np.sum(np.square(first_item-second_item))

    return loss



def show_result(x_edge_matrix_list, p_matrix, prob_result, ground_truth_network):
    loss_list=[]
    for i in range(len(x_edge_matrix_list)):
        cur_matrix=x_edge_matrix_list[i].copy()
        cur_loss=cal_loss(cur_matrix,p_matrix,prob_result)
        loss_list.append(cur_loss)
    max_index=np.argmax(np.array(loss_list))
    max_edge=x_edge_matrix_list[max_index]
    precision, recall, f_score=cal_F1(ground_truth_network,max_edge)

    return precision, recall, f_score



def no_x_likelihood_update_p(p_matrix, s_matrix, prior_network, initial_epsilon, iter_cnt, epsilon_change):
    # cal gradient of p, and update p

    # step 1: cal gradient
    beta, nodes_num = s_matrix.shape
    p_gradient_matrix = np.zeros(p_matrix.shape)

    for i in range(beta):
        for j in range(nodes_num):
            parents = np.where(prior_network[:,j]==1)[0]

            # si=0 term
            gradient_j_zero = np.zeros(nodes_num)
            temp = 1-p_matrix[:,j]*s_matrix[i]
            temp[np.where(temp==0)]=np.inf
            gradient_j_zero = -1*s_matrix[i]/temp*prior_network[:,j]*(1-s_matrix[i,j])
            p_gradient_matrix[:,j]+=gradient_j_zero

            # si=1 term
            gradient_j_one = np.zeros(nodes_num)
            p = p_matrix[:, j].copy()
            s = s_matrix[i].copy()
            A = 1-np.prod(1-p*s)

            if A==0:
                A=np.inf

            p = np.repeat(p[:,np.newaxis], parents.size, axis=1)
            temp = np.arange(parents.size)
            p[parents,temp] = 0     # Fi\vj
            temp_gradient = np.prod(1-p*s[:,np.newaxis], axis=0)*s[parents]/A*s_matrix[i,j]
            gradient_j_one[parents] = temp_gradient.copy()
            p_gradient_matrix[:,j]+=gradient_j_one

            

    # # 调试用
    # # draw hist distribution
    # plt.figure(1000)
    # draw_gradient = p_gradient_matrix[np.where(network_stru==1)].reshape((1,-1)).tolist()
    # plt.hist(draw_gradient,bins=100)
    # plt.show()
    #
    # plt.figure(1001)
    # draw_gradient = np.sort(p_gradient_matrix[np.where(network_stru==1)].reshape((1,-1)))
    # x=np.arange(draw_gradient.shape[1])
    # plt.scatter(x,draw_gradient)
    # print("avg=", np.mean(draw_gradient))
    # plt.show()
    #
    # # 调试用
    # plt.figure(1)
    # draw_p = p_matrix[np.where(network_stru == 1)]
    # plt.hist(draw_p, bins=100)
    # plt.show()

    # step 2: update p
    if epsilon_change:
        epsilon = initial_epsilon/np.sqrt(iter_cnt)
    else:
        epsilon = initial_epsilon

    p_matrix+=epsilon*p_gradient_matrix

    # 调试用
    # plt.figure(2)
    # draw_p = p_matrix[np.where(network_stru == 1)]
    # plt.hist(draw_p, bins=100)
    # plt.show()

    # bound p_matrix
    p_matrix[np.where(p_matrix<0)]=0
    p_matrix[np.where(p_matrix>1)]=1

    # 调试用
    # plt.figure(4)
    # draw_p = p_matrix[np.where(network_stru == 1)]
    # plt.hist(draw_p, bins=100)
    # plt.show()

    return p_matrix



def cal_mae_all(p, infer_p):
    return np.mean(abs(p-infer_p))


def with_x_likelihood_update_x(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_x_sum_threshold,max_delta_x_threshold):
    # cal gradient of x, and update x
    inner_x_cnt=0
    small_value=1e-20
    pre_x_matrix=x_matrix.copy()
    while True:
        inner_x_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        x_gradient_matrix = np.zeros(x_matrix.shape)
    
        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                gradient_j_zero = prior_network[:,j]*(1-s_matrix[i,j])*np.log(temp+small_value)
                x_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                orig_A=A.copy()
                if A==0:
                    A=np.inf
                    
                temp_gradient=prior_network[:,j]*(orig_A-1)*np.log(1-p_matrix[:,j]*s_matrix[i]+small_value)/A*s_matrix[i,j]
                gradient_j_one = temp_gradient.copy()
                x_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update x
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix+=epsilon*x_gradient_matrix

        # bound x_matrix
        x_matrix[np.where(x_matrix<0)]=0
        x_matrix[np.where(x_matrix>1)]=1

        # show x
        max_delta_x=np.max(abs(pre_x_matrix-x_matrix))
        delta_x_sum=np.sum(abs(pre_x_matrix-x_matrix))
        if delta_x_sum<update_x_sum_threshold or max_delta_x<max_delta_x_threshold or inner_x_cnt>30:
            break
    
        pre_x_matrix=x_matrix.copy()

    return x_matrix




def with_x_likelihood_update_p(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_p_sum_threshold, max_delta_p_threshold, p_max_iteration):
    # cal gradient of p, and update p
    inner_p_cnt=0
    small_value=1e-20
    pre_p_matrix=p_matrix.copy()
    while True:
        begin_1=time.time()
        inner_p_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        p_gradient_matrix = np.zeros(p_matrix.shape)

        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                temp[np.where(temp==0)]=np.inf
                gradient_j_zero = -1*s_matrix[i]/temp*prior_network[:,j]*(1-s_matrix[i,j])*x_j
                p_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                
                down=A*(1-s*p)
                down[np.where(down==0)]=np.inf
                up=(-A+1)*x_j*s*s_matrix[i,j]*prior_network[:,j]
                gradient_j_one=up/down
                p_gradient_matrix[:,j]+=gradient_j_one



                # if A==0:
                #     A=np.inf

                # x_ij=x_j[parents].copy()
                # s_il=s[parents].copy()
                # p_ij=p_matrix[parents,j].copy()
                # p = np.repeat(p[:,np.newaxis], parents.size, axis=1)
                # temp = np.arange(parents.size)
                # p[parents,temp] = 0     # Fi\vj
                # temp_gradient = np.prod((1-p*s[:,np.newaxis]+small_value)**x_j.reshape((-1,1)), axis=0)
                # temp_gradient=temp_gradient*x_ij*((1-s_il*p_ij+small_value)**(x_ij-1))*s[parents]/A*s_matrix[i,j]
                # gradient_j_one[parents] = temp_gradient.copy()
                # p_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update p
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        p_matrix+=epsilon*p_gradient_matrix

        # bound p_matrix
        p_matrix[np.where(p_matrix<0)]=0
        p_matrix[np.where(p_matrix>1)]=1

        end_1=time.time()
        print("this iteration time cost=%f"%(end_1-begin_1))

        # show p
        max_delta_p=np.max(abs(pre_p_matrix-p_matrix))
        delta_p_sum=np.sum(abs(pre_p_matrix-p_matrix))
        if delta_p_sum<update_p_sum_threshold or max_delta_p<max_delta_p_threshold or inner_p_cnt>=p_max_iteration:
            break
    
        pre_p_matrix=p_matrix.copy()

    return p_matrix



def show_update_p(ground_truth_network, ground_truth_p, p_matrix, iter_cnt):
    print("inner_p_cnt:%d"%(iter_cnt))
    mae = cal_mae(ground_truth_network, ground_truth_p, p_matrix)
    mse = cal_mse(ground_truth_p, p_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_p, p_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_p, p_matrix)
    print("MAE=%f, MSE=%f, MAE_v2=%f, MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))

    modified_p=modify_p(ground_truth_network, p_matrix)
    modified_mae = cal_mae(ground_truth_network, ground_truth_p, modified_p)
    modified_mse = cal_mse(ground_truth_p, modified_p)
    modified_mae_v2=cal_mae_v2(ground_truth_network, ground_truth_p, modified_p)
    modified_mse_v2=cal_mse_v2(ground_truth_network, ground_truth_p, modified_p)
    print("modified_MAE=%f, modified_MSE=%f, modified_MAE_v2=%f, modified_MSE_v2=%f" % (modified_mae, modified_mse, modified_mae_v2, modified_mse_v2))

    mae_all=cal_mae_all(ground_truth_p, p_matrix)
    print("mae_all=%f"%(mae_all))
 

def show_update_x(ground_truth_network, x_matrix, p_matrix, prob_result, sample_times, iter_cnt):
    print("inner_x_cnt:%d"%(iter_cnt))

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("x_MAE=%f, x_MSE=%f, x_MAE_v2=%f, x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("x_mae_all=%f"%(mae_all))

    x_matrix_list= sample_x(x_matrix, sample_times)
    precision, recall, f1=show_result(x_matrix_list, p_matrix, prob_result, ground_truth_network)
    print("precision=%f,recall=%f,f1=%f"%(precision,recall,f1))



def combine_network(x_edge_matrix_list, p_matrix, prob_result, ground_truth_network, comb_k):
    begin_1=time.time()
    loss_list=[]
    for i in range(len(x_edge_matrix_list)):
        cur_matrix=x_edge_matrix_list[i].copy()
        cur_loss=cal_loss(cur_matrix,p_matrix,prob_result)
        loss_list.append(cur_loss)
    loss_list=np.array(loss_list)
    comb_network=np.zeros(x_edge_matrix_list[0].shape)
    sorted_index=np.argsort(loss_list)
    last_val=np.inf
    comb_cnt=0
    end_1=time.time()
    print("cal x_matrix list score time cost:%f"%(end_1-begin_1))

    sorted_edge_list=[]
    for i in range(len(loss_list)):
        sorted_edge_list.append(x_edge_matrix_list[sorted_index[i]].copy())

    comb_network_list=[] 
    for i in range(loss_list.size-1,-1,-1):
        if loss_list[sorted_index[i]]<last_val:
            last_val=loss_list[sorted_index[i]]
            comb_network+=x_edge_matrix_list[sorted_index[i]]
            comb_network_list.append(comb_network.copy())
            comb_cnt+=1
            if comb_cnt>=comb_k:
                break
    
    for i in range(len(comb_network_list)):
        cur_combine_network=comb_network_list[i]
        cur_combine_network[np.where(cur_combine_network>0)]=1
        comb_network_list[i]=cur_combine_network.copy()
        precision, recall, f_score=cal_F1(ground_truth_network,cur_combine_network)
        print("%d networks:precision=%f,recall=%f,f1=%f"%(i+1,precision,recall,f_score))
        print("edge_num=%d"%(np.sum(comb_network_list[i])))



def show_update_x_combine(ground_truth_network, x_matrix, p_matrix, prob_result, sample_times, iter_cnt, comb_k):
    print("inner_x_cnt:%d"%(iter_cnt))

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("x_MAE=%f, x_MSE=%f, x_MAE_v2=%f, x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("x_mae_all=%f"%(mae_all))

    begin_time=time.time()
    x_matrix_list= sample_x(x_matrix, sample_times)
    end_time=time.time()

    print("sample x_matrix time cost=%f"%(end_time-begin_time))

    begin_2=time.time()
    combine_network(x_matrix_list, p_matrix, prob_result, ground_truth_network,comb_k)
    end_2=time.time()
    print("select combine network time cost=%f"%(end_2-begin_2))


def with_x_likelihood_update_x_combine(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_x_sum_threshold, max_delta_x_threshold, x_max_iteration):
    # cal gradient of x, and update x
    inner_x_cnt=0
    small_value=1e-20
    pre_x_matrix=x_matrix.copy()
    while True:
        begin_1=time.time()
        inner_x_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        x_gradient_matrix = np.zeros(x_matrix.shape)
    
        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                gradient_j_zero = prior_network[:,j]*(1-s_matrix[i,j])*np.log(temp+small_value)
                x_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                orig_A=A.copy()
                if A==0:
                    A=np.inf
                    
                temp_gradient=prior_network[:,j]*(orig_A-1)*np.log(1-p_matrix[:,j]*s_matrix[i]+small_value)/A*s_matrix[i,j]
                gradient_j_one = temp_gradient.copy()
                x_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update x
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix+=epsilon*x_gradient_matrix

        # bound x_matrix
        x_matrix[np.where(x_matrix<0)]=0
        x_matrix[np.where(x_matrix>1)]=1

        end_1=time.time()
        print("this iteration time cost=%f"%(end_1-begin_1))

        # show x
        max_delta_x=np.max(abs(pre_x_matrix-x_matrix))
        delta_x_sum=np.sum(abs(pre_x_matrix-x_matrix))
        if delta_x_sum<update_x_sum_threshold or max_delta_x<max_delta_x_threshold or inner_x_cnt>=x_max_iteration:
            break
    
        pre_x_matrix=x_matrix.copy()

    return x_matrix




def show_init_x(ground_truth_network, x_matrix):

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("init_x_MAE=%f, init_x_MSE=%f, init_x_MAE_v2=%f, init_x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("init_x_mae_all=%f"%(mae_all))



def kmeans_zero(data):
    data_num=data.size
    
    center_0=0
    for i in range(data_num):
        if data[i]>0:
            center_1=data[i]
            break
    
    max_iteration=300
    stop_threshold=1e-5
    cur_iter=0
    
    label_distribution = -1*np.ones(data_num)
    pre_center_1=center_1
    while True:
        cur_iter+=1
        for i in range(data_num):
            dist_0=abs(data[i]-center_0)
            dist_1=abs(data[i]-center_1)
            if dist_0<dist_1:
                label_distribution[i]=0
            else:
                label_distribution[i]=1
        
        # update center
        center_1=np.mean(data[np.where(label_distribution==1)])
        if abs(center_1-pre_center_1)<stop_threshold or cur_iter>max_iteration:
            break
        pre_center_1=center_1
        
    return label_distribution




def weighted_mi_prune_zero(record_states, prune_choice):
    # prune_choice = 0  mi   ;  prune_choice = 1 imi

    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[0,0]+=(1-record_states[result_index,j])*(1-record_states[result_index,k])
                state_mat[0,1]+=(1-record_states[result_index,j])*record_states[result_index,k]
                state_mat[1,0]+=record_states[result_index,j]*(1-record_states[result_index,k])
                state_mat[1,1]+=record_states[result_index,j]*record_states[result_index,k]

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

            if prune_choice==0:
                MI[j, k] = M00 + M11 +M10 + M01
            else:
                MI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            MI[k, j] = MI[j, k]

    # Kmeans 聚类
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    label_pred = kmeans_zero(tmp_MI)
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    return prune_network



def load_data_memetracker(graph_path_list, result_path, cascades_path, beta):
    # load data: groundtruth network, complete_record
    gt_network_list = []
    for i in range(len(graph_path_list)):
        nodes_num = 0
        edge_flag = False
        with open(graph_path_list[i], 'r') as f:
            for line in f:
                if not edge_flag and line != '\n':
                    nodes_num += 1
                elif line == '\n':
                    edge_flag = True
                    groundtruth_network = np.zeros((nodes_num, nodes_num))
                elif edge_flag:
                    edge_info = re.split('[\t\n]', line)
                    node1 = int(edge_info[0])
                    node2 = int(edge_info[1])
                    groundtruth_network[node1, node2] = 1
        for i in range(nodes_num):
            groundtruth_network[i, i] = 0
        gt_network_list.append(groundtruth_network.copy())

    diffusion_results = np.zeros((beta, nodes_num))
    dif_res_cnt = 0
    dif_res_flag = False
    with open(result_path, 'r') as f2:
        for line in f2:
            if dif_res_cnt >= beta:
                break
            if not dif_res_flag and line == '\n':
                dif_res_flag = True
            elif dif_res_flag and line[0] == ';':
                dif_res_info = re.split('[;\n]', line[1:])
                for i in range(len(dif_res_info) - 1):
                    diffusion_results[dif_res_cnt, int(dif_res_info[i])] = 1
                dif_res_cnt += 1

    cascades = np.ones((beta, nodes_num)) * np.inf
    cas_cnt = 0
    cas_flag = False
    with open(cascades_path, 'r') as f3:
        for line in f3:
            if cas_cnt >= beta:
                break
            if not cas_flag and line == '\n':
                cas_flag = True
            elif cas_flag and line[0] == ';':
                cas_info = re.split('[;,\n]', line)
                for i in range(1, len(cas_info) - 1, 2):
                    domain_id = int(cas_info[i])
                    utime = float(cas_info[i + 1])
                    cascades[cas_cnt, domain_id] = utime
                cas_cnt += 1
    cascades = cascades - np.min(cascades, axis=1).reshape((-1, 1))
    return gt_network_list, diffusion_results, cascades



def meme_sample(prob_result, sample_times):
    result_list=[]
    for i in range(sample_times):
        cur_sample=np.zeros(prob_result.shape)
        sample = np.random.rand(*prob_result.shape)
        one_index = np.where(sample < prob_result)
        cur_sample[one_index] = 1
        result_list.append(cur_sample.copy())

    return result_list



def hsic_construct_list_with_cnt(results_list, cascades,delta_t, coexist_bound, use_time, topk, sample_times):
    network_list = []
    for i in range(sample_times):
        constructed_network = hsic_construct_with_cnt(results_list[i], cascades,delta_t, coexist_bound, use_time, topk)
        network_list.append(constructed_network)

    return network_list


def hsic_construct_with_cnt(diffusion_result, cascades,delta_t, coexist_bound, use_time, topk):
    _, pruning_matrix=cnt_par(diffusion_result, cascades, delta_t, coexist_bound, use_time, topk)

    return pruning_matrix


def cnt_par(diffusion_results, cascades, delta_t, coexist_bound, use_time, topk):

    beta,nodes_num = diffusion_results.shape
    co_exist_matrix=np.zeros((nodes_num,nodes_num))
    temp_cascades=cascades.copy()
    for i in range(beta):
        inf_nodes = np.where(diffusion_results[i]==1)[0]
        if use_time:
            for j in inf_nodes:
                add_nodes = np.where((temp_cascades[i]-temp_cascades[i,j]>=-delta_t) & (temp_cascades[i]-temp_cascades[i,j]<=0))[0]
                co_exist_matrix[add_nodes,j]+=1
        else:
            for j in inf_nodes:
                co_exist_matrix[inf_nodes,j]+=1

    cnt_matrix = np.zeros(co_exist_matrix.shape)
    cnt_matrix[np.where(co_exist_matrix>=coexist_bound)]=1

    for i in range(nodes_num):
        cnt_matrix[i,i]=0

    if topk>0:
        for i in range(nodes_num):
            candidate_par=np.where(cnt_matrix[:,i]==1)[0]
            if len(candidate_par)<=topk:
                continue
            coexist_cnt=co_exist_matrix[candidate_par,i]
            temp_list=[]
            for j in range(len(candidate_par)):
                temp_list.append((candidate_par[j],coexist_cnt[j]))
            sorted_par=sorted(temp_list, reverse=True, key=lambda cpar: cpar[1])
            topk_par=sorted_par[:topk]
            res_topk_par=[par[0] for par in topk_par]
            cnt_matrix[:,i]=0
            cnt_matrix[res_topk_par,i]=1


    # print("par num=")
    # print(np.sum(cnt_matrix,axis=0))

    return co_exist_matrix, cnt_matrix
    

def combine_network_meme(network_list):
    # 将sample_times个network拼成一个
    nodes_num = network_list[0].shape[0]
    comb_network = np.zeros((nodes_num,nodes_num))
    for network in network_list:
        comb_network = comb_network + network

    comb_network[comb_network >= 1] =1

    return comb_network



def soft_coexist(prob_result):
    beta, nodes_num=prob_result.shape
    coexist_matrix=np.zeros((nodes_num,nodes_num))
    for k in range(beta):
        for i in range(nodes_num):
            temp_result=prob_result[k,:].copy()
            temp_result*=prob_result[k,i]
            coexist_matrix[i,:]+=temp_result

    # Kmeans 聚类
    tmp_coexist = coexist_matrix.reshape((-1, 1)).copy()

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_coexist)
    label_pred = estimator.labels_
    temp_0 = tmp_coexist[label_pred == 0]
    temp_1 = tmp_coexist[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(coexist_matrix>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    return prune_network


def Pearson_prune(record_states,ground_truth_network):
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    IMI = np.zeros((nodes_num, nodes_num))

    IMI = np.corrcoef(record_states.T)
    # for j in range(nodes_num):
    #     for k in range(nodes_num):
    #         if j > k:
    #             continue
    #         mat = np.vstack((record_states[:,j].reshape(1,-1), record_states[:,k].reshape(1,-1)))
    #         temp = np.corrcoef(mat)
    #         IMI[j, k] = temp[0,1]
    #         IMI[k, j] = IMI[j, k]

    # Kmeans 聚类
    # IMI = np.abs(IMI)
    # IMI[np.where(IMI<0)] = 0
    # for i in range(nodes_num):
    #     IMI[i,i] = -1
    tmp_IMI = IMI.reshape((-1, 1))
    tmp_IMI = tmp_IMI[np.where(tmp_IMI>0)].reshape((-1,1)) # 在这里切的时候只考虑大于0的值（跟tends的java版本一致）
                                                             # 只考虑正数的情况和一刀切，考虑0值的情况在beta=100(120),150的时候有所差别，其余无明显差别

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_IMI)
    label_pred = estimator.labels_
    temp_0 = tmp_IMI[label_pred == 0]
    temp_1 = tmp_IMI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    # draw_values(ground_truth_network, IMI)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(IMI>tau)] = 1
    for i in range(nodes_num):
        prune_network[i,i] = 0

    return prune_network
    

def new_Pearson_prune(record_states,ground_truth_network):
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    IMI = np.zeros((nodes_num, nodes_num))

    IMI = np.corrcoef(record_states.T)
    for j in range(nodes_num):
        for k in range(nodes_num):
            if j > k:
                continue
            # mat = np.vstack((record_states[:,j].reshape(1,-1), record_states[:,k].reshape(1,-1)))
            # temp = np.corrcoef(mat)
            cov = np.mean(record_states[:,j]*record_states[:,k])-np.mean(record_states[:,j])*np.mean(record_states[:,k])
            varxj = np.mean(record_states[:,j]*record_states[:,j]) - np.mean(record_states[:,j])*np.mean(record_states[:,j])
            varxk = np.mean(record_states[:,k]*record_states[:,k]) - np.mean(record_states[:,k])*np.mean(record_states[:,k])
            coe = cov / np.sqrt(varxj * varxk)
            IMI[j, k] = coe
            IMI[k, j] = IMI[j, k]

    # Kmeans 聚类
    # IMI = np.abs(IMI)
    # IMI[np.where(IMI<0)] = 0
    # for i in range(nodes_num):
    #     IMI[i,i] = -1
    tmp_IMI = IMI.reshape((-1, 1))
    tmp_IMI = tmp_IMI[np.where(tmp_IMI>0)].reshape((-1,1)) # 在这里切的时候只考虑大于0的值（跟tends的java版本一致）
                                                             # 只考虑正数的情况和一刀切，考虑0值的情况在beta=100(120),150的时候有所差别，其余无明显差别

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_IMI)
    label_pred = estimator.labels_
    temp_0 = tmp_IMI[label_pred == 0]
    temp_1 = tmp_IMI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    # draw_values(ground_truth_network, IMI)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(IMI>tau)] = 1
    for i in range(nodes_num):
        prune_network[i,i] = 0

    return prune_network
        
