import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

''' Get Similarity Matrix '''

def get_eu_dist_matrix(eivec_matrix: np.ndarray):
    node_num = eivec_matrix.shape[0]
    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                dist_matrix[i][j] = 0
                continue
            if i > j:
                dist_matrix[i][j] = dist_matrix[j][i]
                continue
            dist_matrix[i][j] = np.sqrt(np.sum(np.square(eivec_matrix[i] - eivec_matrix[j])))   # 欧式距离
    return dist_matrix


def get_homop_matrix(eivec_matrix: np.ndarray):
    node_num = eivec_matrix.shape[0]
    similarity_matrix = np.zeros((node_num, node_num))

    # 对矩阵预处理，避免负数
    min_value = np.min(eivec_matrix)
    if min_value < 0:
        eivec_matrix += -min_value

    for i in range(node_num):
        for j in range(node_num):
            vector_i = eivec_matrix[i]
            vector_j = eivec_matrix[j]
            similarity_matrix[i][j] = np.sum(
                ((np.minimum(vector_i, vector_j) + 1e-6) / (np.maximum(vector_i, vector_j) + 1e-6))
                * (np.abs(vector_i + vector_j) / np.sum(vector_i + vector_j)))
    return similarity_matrix

def cal_Ws(sim_matrix, index_subset):
    node_count = len(index_subset)
    Bs = np.ones((node_count + 1, node_count + 1), dtype=np.float)
    Bs[0, 0] = 0
    Bs[1:, 1:] = sim_matrix[index_subset][:, index_subset]

    return np.power(-1, node_count) * np.linalg.det(Bs)


def get_homop_sim_matrix(eivec_matrix: np.ndarray):
    node_num = eivec_matrix.shape[0]
    similarity_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            vector_i = eivec_matrix[i]
            vector_j = eivec_matrix[j]
            similarity_matrix[i][j] = np.sum(
                ((np.minimum(vector_i, vector_j) + 1e-6) / (np.maximum(vector_i, vector_j) + 1e-6))
                * (np.abs(vector_i + vector_j) / np.sum(vector_i + vector_j)))
    return similarity_matrix


def get_cosine_sim_matrix(eivec_matrix: np.ndarray):
    node_num = eivec_matrix.shape[0]
    similarity_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            vector_i = eivec_matrix[i]
            vector_j = eivec_matrix[j]
            similarity_matrix[i][j] = np.dot(vector_i, vector_j) / (
                    np.linalg.norm(vector_i) * np.linalg.norm(vector_j))
    return similarity_matrix


def get_man_sim_matrix(eivec_matrix: np.ndarray):
    node_num = eivec_matrix.shape[0]
    similarity_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            vector_i = eivec_matrix[i]
            vector_j = eivec_matrix[j]
            similarity_matrix[i][j] = 1 - (np.sum(np.abs(vector_i - vector_j)) / node_num)
    return similarity_matrix


def get_eu_sim_matrix(eivec_matrix: np.ndarray):
    dist_matrix = get_eu_dist_matrix(eivec_matrix)
    similarity_matrix = np.exp(-dist_matrix / np.mean(dist_matrix))  # 求相似度矩阵
    return similarity_matrix, dist_matrix

 
def cluster_hier(eivec_matrix: np.ndarray, SIM_MODE="Sim"):
    if SIM_MODE == "Sim":
        sim_matrix = get_homop_matrix(eivec_matrix)
    elif SIM_MODE == "COSINE":
        sim_matrix = get_cosine_sim_matrix(eivec_matrix)
    elif SIM_MODE == "MAN":
        sim_matrix = get_man_sim_matrix(eivec_matrix)
    elif SIM_MODE == "EU":
        sim_matrix, _ = get_eu_sim_matrix(eivec_matrix)
    elif SIM_MODE == "NONE":
        sim_matrix = eivec_matrix
    else:
        raise ValueError("Sim Mode Doesn't Exist!")
    np.fill_diagonal(sim_matrix, 0)           # A的对角线元素为0

    node_count = sim_matrix.shape[0]
    subset_count = node_count
    T = np.zeros((node_count, node_count), dtype=np.float)      # Tij表示将集合i与集合j合并后的Ws

    # 初始化矩阵T
    for i in range(node_count):
        for j in range(i + 1, node_count):
            index_subset = [i, j]
            T[i][j] = cal_Ws(sim_matrix, index_subset)
            T[j][i] = T[i][j]

    # 迭代合并集合
    subsets = [[i] for i in range(node_count)]
    while np.max(T) > 0 and subset_count > 3:
        i, j = np.unravel_index(np.argmax(T), T.shape)
        # 合并集合
        subsets[i] += subsets[j]
        subsets[j].clear()
        subset_count -= 1
        # 更新矩阵T
        T[j, :] = np.NINF
        T[:, j] = np.NINF
        for k in range(node_count):
            if T[i, k] == np.NINF:      # 该行/列已经被舍弃
                continue
            index_subset = subsets[i] + subsets[k]
            T[i, k] = cal_Ws(sim_matrix, index_subset)
            T[k, i] = T[i, k]

    # 将结果映射为A
    A = np.zeros((node_count, subset_count), dtype=np.int)
    subset_index = 0
    for s in range(0, len(subsets)):
        if len(subsets[s]) == 0:
            continue
        for node in subsets[s]:
            A[node][subset_index] = 1
        subset_index += 1

    return A

