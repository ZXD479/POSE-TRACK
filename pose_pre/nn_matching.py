"""
Modified code from https://github.com/nwojke/deep_sort
"""

import copy
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from pose_pre.pose_trsanformerv2 import get_pose_distance
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from sklearn.linear_model import Ridge
def pose_similarity(history_pose, candidate_pose):
    # history_pose: (10, 226), candidate_pose: (226,)
    sim = np.mean([1 - cosine(frame, candidate_pose) for frame in history_pose])
    return sim
def compute_distance_matrix(track_pose, detect_pose):  
    A = track_pose[:, :-3]
    B = detect_pose[:, :-3]

    cost_matrix = euclidean_distances(A, B)

    # 归一化到 [0, 1] 范围
    min_val, max_val = cost_matrix.min(), cost_matrix.max()
    normalized_cost_matrix = (cost_matrix - min_val) / (max_val - min_val)

    #print(normalized_cost_matrix)

    return normalized_cost_matrix
def compute_cost_matrix(A, B):  

    # 检查A和B的列数是否相同  
    if A.shape[1] != B.shape[1]:  
        raise ValueError("A和B的列数必须相同")  
      
    # 初始化代价矩阵  
    cost_matrix = np.zeros((A.shape[0], B.shape[0]))  
      
    # 计算余弦相似度并转换为代价  
    for i in range(A.shape[0]):  
        for j in range(B.shape[0]):  
            similarity = np.dot(A[i], B[j]) / (np.linalg.norm(A[i]) * np.linalg.norm(B[j]))  
            cost = 1 - similarity  
            cost_matrix[i, j] = cost  
      
    return cost_matrix  
def weighted_euclidean_distance(trajectory1, trajectory2, weights):
    """
    计算加权欧氏距离
    :param trajectory1: 第一个轨迹（例如最新一帧的某个人的轨迹）
    :param trajectory2: 第二个轨迹（例如历史7帧的轨迹）
    :param weights: 每个维度的权重
    :return: 加权欧氏距离
    """
    # 确保轨迹为1x229的形状
    trajectory1 = np.asarray(trajectory1).reshape(1, -1)
    trajectory2 = np.asarray(trajectory2).reshape(1, -1)
    
    # 计算加权欧氏距离
    distance = np.sqrt(np.sum(weights * (trajectory1 - trajectory2) ** 2))
    return distance

def calculate_similarity(history_trajectories, latest_frame_trajectories, weights):
    """
    计算历史轨迹与最新一帧的多个人的轨迹的相似度
    :param history_trajectories: 历史轨迹，形状为 (7, 229)
    :param latest_frame_trajectories: 最新一帧的多个人轨迹，形状为 (10, 229)
    :param weights: 每个维度的权重
    :return: 返回每个目标与历史轨迹的相似度
    """
    num_people = latest_frame_trajectories.shape[0]  # 最新一帧的目标人数
    similarity_scores = []

    # 对每个最新目标进行相似度计算
    for i in range(num_people):
        latest_trajectory = latest_frame_trajectories[i]
        distances = []
        
        # 遍历历史轨迹的每一帧，计算距离
        for j in range(history_trajectories.shape[0]):
            history_trajectory = history_trajectories[j]
            distance = weighted_euclidean_distance(latest_trajectory, history_trajectory, weights[j])
            distances.append(distance)
        
        # 将每个目标的历史轨迹的距离输出
        similarity_scores.append(distances)
    
    return similarity_scores
def _pdist_l2(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`.""" 
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2
def min_max(array):
    if array.size > 0:  
        # 计算Min-Max归一化所需的最小值和最大值  
        min_val = np.min(array)  
        max_val = np.max(array)  
        
        # 应用Min-Max归一化公式  
        normalized_array = (array - min_val) / (max_val - min_val)  
    return normalized_array
def adaptive_normalized_fusion(app_cost, pose_cost):
    if app_cost.size:
        app_norm = app_cost / (app_cost.max() + 1e-8)
        pose_norm = pose_cost / (pose_cost.max() + 1e-8)
        return app_norm + pose_norm
    else:
        return pose_cost + app_cost


def _pdist(cfg, a, b, i, track_end_time, frame_id):#

    if not hasattr(_pdist, "ridge_model"):
        _pdist.ridge_model = Ridge(alpha=1.0, fit_intercept=True)
    cfg.phalp.distance_type="PL"
    cfg.phalp.predict="PL"
    a_pose    = []
    for i_ in range(len(a)):
        a_pose.append(a[i_][0]); #1,229

    if b[1] is not None:#soccer

        b_pose,  b_appe ,b_iou= b[0], b[1], b[2]#7,99   7,229
        a_pose, b_pose  = np.asarray(a_pose), copy.deepcopy(np.asarray(b_pose)) 
        
        n_appce = b_appe[i]

        track_pose      = a_pose
        detect_pose     = b_pose
        # if frame_id > 7:

        #     X = np.arange(frame_id - 7, frame_id).reshape(-1, 1)
        #     _pdist.ridge_model.fit(X, track_pose.squeeze(0))
        #     next_x = np.array([[frame_id]])
        #     predicted_pose = _pdist.ridge_model.predict(next_x)
        #     pose_distance = cdist(predicted_pose, detect_pose, 'cosine') / 2.0 
        # else:

        #pose_distance = cdist(track_pose[:,-1,:], detect_pose, 'cosine') / 2.0 
        pose_distance = cdist(track_pose, detect_pose, 'cosine') / 2.0 
        pose_distance[pose_distance>0.25] = 1
        #n_appce = n_appce * 5.0

        ruv2                = ((n_appce)*0.7) + (pose_distance*0.3)
        #ruv2                = pose_distance
        #ruv2 = np.minimum(n_appce, pose_distance)
        ruv2                = np.nan_to_num(ruv2)
        return ruv2    
    else:
        b_pose,b_iou= b[0],b[2]
        a_pose, b_pose  = np.asarray(a_pose), copy.deepcopy(np.asarray(b_pose))
        

        track_pose      = a_pose
        detect_pose     = b_pose
        pose_distance1 = cdist(track_pose, detect_pose, 'cosine') / 2.0    
        pose_distance1[pose_distance1>0.25] = 1

        pose_distance1 = pose_distance1.reshape(-1)


        ruv2                = (pose_distance1)
        ruv2                = np.nan_to_num(ruv2)
        return ruv2        

    
def _nn_euclidean_distance_min(cfg, x, y, i, track_end_time, frame_id):

    distances_a = _pdist(cfg, x, y, i, track_end_time, frame_id)
    
    return distances_a



class NearestNeighborDistanceMetric(object):

    def __init__(self, cfg, matching_threshold, budget=None):
        
        self.cfg                = cfg
        self._metric            = _nn_euclidean_distance_min
        self.matching_threshold = matching_threshold
        self.budget             = budget
        self.samples            = {}
        
        
    def partial_fit(self, pose_features, targets, active_targets):#标记

        for pose_feature,  target in zip(pose_features, targets):
            self.samples.setdefault(target, []).append([pose_feature])
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, detection_features, targets, track_end_time, frame_id):

        cost_matrix_a = np.zeros((len(targets), len(detection_features[0])))
        for i, target in enumerate(targets):
            cost_matrix_a[i, :] = self._metric(self.cfg, self.samples[target], detection_features, i, track_end_time[i], frame_id)
        return cost_matrix_a