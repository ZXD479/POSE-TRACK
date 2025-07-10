# import sys  
# sys.path.append('MixViT')
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resized_crop, normalize
import math
from torch.utils.tensorboard import SummaryWriter
from pose_pre import nn_matching
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
from MixViT.lib.models.mixformer_vit import build_mixformer_deit
from MixViT.lib.train.data.processing import MixformerProcessing as MP
from MixViT.lib.train.data.transforms import Transform, ToTensor, Normalize
import MixViT.lib.train.data.processing_utils as prutils
import MixViT.lib.train.admin.settings as ws_settings
import importlib
from MixViT.lib.train.base_functions import update_settings
import MixViT.lib.train.data.transforms as tfm
from typing import List, Union, Tuple
from pose_pre import linear_assignment
from pose_pre.track import Track 
from sklearn.linear_model import Ridge
import scipy.stats as stats
from pose_pre.nn_matching import adaptive_normalized_fusion
import copy
def map_to_range(x_old, old_min=0, old_max=500, new_min=-5, new_max=20):

    # 计算线性变换的系数
    a = (new_max - new_min) / (old_max - old_min)
    b = new_min - a * old_min
    
    # 应用线性变换
    x_new = a * x_old + b
    
    return x_new
def min_max(array):
    if array.size > 0:  
        min_val = np.min(array)  
        max_val = np.max(array)  

        normalized_array = (array - min_val) / (max_val - min_val)  
    return normalized_array
old_settings = np.seterr(divide='ignore', invalid='ignore') 

class STrack_eiou_pose(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, keypoints=None, feat_history=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.last_tlwh = self._tlwh

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.features = []
        self.times = []
        self.alpha = 0.9
        self.keypoints = keypoints


    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)


    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    
    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        self.last_tlwh = new_track.tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.last_tlwh = new_tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def last_tlbr(self):

        ret = self.last_tlwh.copy()#应该在这里改滤波
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)





class Deep_EIoU_soccer_pose(object):
    def __init__(self, cfg, args, pose_predictor=None, frame_rate=30):

        self.tracked_stracks = []  
        self.lost_stracks = []  
        self.removed_stracks = []  
        BaseTrack.clear_count()
        self.cfg = cfg
        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.tracks = []
        self._next_id = 1
        self.pose_predictor = pose_predictor
        self.max_age    = 30
        self.A_dim = 4096
        self.P_dim = 4096
        self.L_dim = 99
        self.n_init = 5
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.mode = ''
    
    
    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  
        self.lost_stracks = [] 
        self.removed_stracks = []  

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh
        self.tracks = []
        self._next_id = 1


  
    def _match(self, detections, tracked_stracks, appce=None, pose_mask=None, ious_dists=None):

        def gated_metric(tracks, dets, track_indices, detection_indices, appce, iou_dists):

            #loca_emb          = np.array([dets[i]['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i]['pose'] for i in detection_indices])
            appce             = appce
            ious_dists        = iou_dists
            targets           = np.array([tracked_stracks[i].track_id for i in track_indices])
            track_end_time    = [tracked_stracks[i].end_frame for i in track_indices]
            cost_matrix       = self.metric.distance([pose_emb, appce, ious_dists], targets, track_end_time, self.frame_id)
            return cost_matrix

        #matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, joint_stracks(self.tracked_stracks, self.lost_stracks), detections, confirmed_tracks)
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, tracked_stracks , detections, appce, ious_dists)
        # track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        # detect_gt  = [d['ground_truth'] for i, d in enumerate(detections)]

        # track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        # detect_idt = [i for i, d in enumerate(detections)]
        
        if appce is not None:    
            if appce.size == 0:
                cost_pose_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
            else:         
                if len(cost_matrix) == 1:
                    cost_pose_matrix = cost_matrix
                else:
                    cost_pose_matrix = min_max(cost_matrix)


            #cost_pose_matrix = cost_matrix
        else:
            if not len(cost_matrix):
                cost_pose_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
            else: 
                if len(cost_matrix) == 1:
                    cost_pose_matrix = cost_matrix
                else:
                    cost_pose_matrix = min_max(cost_matrix)



        return matches, unmatched_tracks, unmatched_detections, cost_pose_matrix
                                          #loca=l_pred[p_id] if("L" in features) else None)
                
    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1  


    def update(self, asd, output_results, embedding):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
  
            bboxes = output_results

            scores = np.full((bboxes.shape[0]), 1.0, dtype=np.float64)

            lowest_inds = scores > self.track_low_thresh#可能是0.4
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            if self.args.with_pose:
                asd_all = asd
                asd = [item for item, i in zip(asd, remain_inds) if i == True]
            
            if self.args.with_reid:

                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]


        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []      

        if len(dets) > 0:
            '''Detections'''

            if self.args.with_reid:
                detections = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s, f) for
                            (tlbr, s, f) in zip(dets, scores_keep, features_keep)]

            else:
                detections = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = [] 
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        #oppo = self.tracks
        # Associate with high score detection boxes
        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):
            
            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

            ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)
            pose_mask = (ious_dists < 0.25)
            #if self.args.with_keypoints:

            if self.args.with_reid:
                
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                _, _, _, statistics = self._match(asd, strack_pool, emb_dists, pose_mask, ious_dists)

                dists = np.minimum(ious_dists, statistics)
                #dists = 0.7 * ious_dists + 0.3 * statistics  


            else:
                if self.args.with_pose:
                    _, _, _, statistics = self._match(asd, strack_pool, None, pose_mask, ious_dists)
                    
                    stat_weight = 0.3 * np.ones((len(strack_pool), 1))  # shape: (num_tracks, 1)

                    # 遍历所有轨迹，调整处于 lost 状态的 statistic 权重
                    for i, track in enumerate(strack_pool):
                        if track.state == 3:  # lost 状态
                            stat_weight[i] = 0.15  # 降低权重

                    # 构造 dists
                    dists = (1 - stat_weight) * ious_dists + stat_weight * statistics                    
                    
                    
                    #dists = 0.7 * ious_dists + 0.3 * statistics
                    #dists = ious_dists * statistics


                    #adaptive_normalized_fusion(ious_dists, statistics)
                    #dists=np.maximum(ious_dists, statistics)
                    #np.minimum(ious_dists, statistics)
                    #dists=np.sqrt(ious_dists * statistics)
                else:
                #dists = statistics
                    dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

            if self.args.with_pose:
                index = [strack_pool[i[0]].track_id for i in matches]
                u_index = [strack_pool[i].track_id for i in u_track]
                indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]
                u_indices = [i for i, track in enumerate(self.tracks) if track.track_id in u_index]
                for (track_idx, detection_idx), oppo in zip(matches,indices):
                    self.tracks[oppo].update(asd[detection_idx], detection_idx, 0)
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            
            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
            #if self.args.with_reid:
            asd = [asd[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            
            if self.args.with_reid:
                features_second = embedding[inds_second]
            if self.args.with_pose:
                asd_second = [item for item, i in zip(asd_all, inds_second) if i == True]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections_second = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        ious_dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.6)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        pose_mask = (ious_dists < 0.25)
        if self.args.with_reid:
            
            emb_dists = matching.embedding_distance(r_tracked_stracks, detections_second) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            _, _, _, statistics = self._match(asd_second, r_tracked_stracks, emb_dists, pose_mask, ious_dists)
            dists = statistics
            #dists = 0.8*ious_dists + 0.2 * statistics
        else:
            #dists = ious_dists  
            if self.args.with_pose:       
                stat_weight = 0.3 * np.ones((len(r_tracked_stracks), 1))   
                _, _, _, statistics = self._match(asd_second, r_tracked_stracks, None, pose_mask, ious_dists)

                for i, track in enumerate(r_tracked_stracks):
                    if track.state == 3:  # lost 状态
                        stat_weight[i] = 0.15  # 降低权重

                # 构造 dists
                dists = (1 - stat_weight) * ious_dists + stat_weight * statistics    
                #dists = 0.7 * ious_dists + 0.3 * statistics
            else:
                dists = ious_dists
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.8)#要不要在这里也价加个姿态呢？
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        if self.args.with_pose:
            index = [r_tracked_stracks[i[0]].track_id for i in matches]
            u_index = [r_tracked_stracks[i].track_id for i in u_track]
            indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]
            
            for (track_idx, detection_idx), oppo in zip(matches,indices):#77.5应该也没这个东西
                self.tracks[oppo].update(asd_second[detection_idx], detection_idx, 0)#这里不对，二次匹配的时候带有预测不应该是这样
            
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])#这里self.track已经包括他了，因为最后有一个包括的动作，所以这里是没错的
        for it in u_unconfirmed:#为什么self.tracks多呢，是因为最前面包括了未激活的，但对未激活的没做什么操作
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        if self.args.with_pose:
            for detection_idx in u_detection:
                if asd[detection_idx]['conf'] < self.new_track_thresh:
                    continue
                self._initiate_track(asd[detection_idx], detection_idx)#在这里，第一针检测到的新物体会被添加进track
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""

        for track in self.lost_stracks:

            if self.frame_id - track.end_frame >  map_to_range(track.end_frame - track.start_frame) + 60:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """



        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        
        if self.args.with_pose:
            track_ids = set(track.track_id for track in joint_stracks(self.tracked_stracks, self.lost_stracks))

            self.tracks = [track for track in self.tracks if track.track_id in track_ids]
                        
            track2_dict = {track.track_id: track for track in self.tracks}

            self.tracks = [track2_dict[track.track_id] for track in joint_stracks(self.tracked_stracks, self.lost_stracks) if track.track_id in track2_dict]

            pose_features,targets = [], []

            active_targets = [t.track_id for t in joint_stracks(self.tracked_stracks,self.lost_stracks)]
            for track in self.tracks:

                pose_features += [track.track_data['history'][-1]['pose']]  
                targets       += [track.track_id]
                
            self.metric.partial_fit(np.asarray(pose_features), np.asarray(targets), active_targets)

        return output_stracks
        
            
class Pose_Track(object):
    def __init__(self, cfg, args, pose_predictor=None, frame_rate=30):

        self.tracked_stracks = []  
        self.lost_stracks = [] 
        self.removed_stracks = []  
        BaseTrack.clear_count()
        self.cfg = cfg
        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.tracks = []
        self._next_id = 1
        self.pose_predictor = pose_predictor
        self.max_age    = 30
        self.A_dim = 4096
        self.P_dim = 4096
        self.L_dim = 99
        self.n_init = 5
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.mode = ''
    
    
    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  
        self.lost_stracks = []  
        self.removed_stracks = [] 

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh
        self.tracks = []
        self._next_id = 1

 
    def _match(self, detections, tracked_stracks, appce=None, ious_dists=None):

        def gated_metric(tracks, dets, track_indices, detection_indices, appce, ious_dists):

            pose_emb          = np.array([dets[i]['pose'] for i in detection_indices])
            appce             = appce
            ious_dists        = ious_dists
            targets           = np.array([tracked_stracks[i].track_id for i in track_indices])
            track_end_time    = [tracked_stracks[i].end_frame for i in track_indices]
            cost_matrix       = self.metric.distance([pose_emb, appce, ious_dists], targets, track_end_time, self.frame_id)
            return cost_matrix

        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, tracked_stracks , detections, appce, ious_dists)
        
        if appce is not None:    
            if appce.size == 0:
                cost_pose_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
                #cost_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
            else:         
                if len(cost_matrix) == 1:
                    cost_pose_matrix = cost_matrix
                else:
                    cost_pose_matrix = min_max(cost_matrix)
                    #cost_pose_matrix = cost_matrix

        else:
            if not len(cost_matrix):
                cost_pose_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
            else: 
                if len(cost_matrix) == 1:
                    cost_pose_matrix = cost_matrix
                else:
                    cost_pose_matrix = min_max(cost_matrix)

        return matches, unmatched_tracks, unmatched_detections, cost_pose_matrix                            
        #return matches, unmatched_tracks, unmatched_detections, cost_matrix         
    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1  


    def update(self, asd, output_results, embedding):
        
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:,4]
                bboxes = output_results[:, :4]  # x1y1x2y2
            elif output_results.shape[1] == 7:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            else:
                raise ValueError('Wrong detection size {}'.format(output_results.shape[1]))

            lowest_inds = scores > self.track_low_thresh#可能是0.4
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            asd_all = asd
            asd = [item for item, i in zip(asd, remain_inds) if i == True]
            
            if self.args.with_reid:
                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []      

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s, f) for
                            (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):
            
            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

            ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.args.with_reid:            
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                _, _, _, statistics = self._match(asd, strack_pool, emb_dists, ious_dists)
                dists = np.minimum(ious_dists, statistics)
                #dists = np.minimum(emb_dists, statistics)
                #dists = statistics
                #dists = 0.65*statistics + 0.35*ious_dists
                # if len(dists) == 1 or len(dists) == 0 or emb_dists.size == 0:
                #     dists = dists
                # else:
                #     dists = min_max(dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

            index = [strack_pool[i[0]].track_id for i in matches]
            indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]
        
            
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)


            for (track_idx, detection_idx), oppo in zip(matches,indices):
                self.tracks[oppo].update(asd[detection_idx], detection_idx, 0)#这里不对，二次匹配的时候带有预测不应该是这样

            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
            asd = [asd[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            asd_second = [item for item, i in zip(asd_all, inds_second) if i == True]
            if self.args.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections_second = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack_eiou_pose(STrack_eiou_pose.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        ious_dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.6)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:    
            emb_dists = matching.embedding_distance(r_tracked_stracks, detections_second) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            _, _, _, statistics = self._match(asd_second, r_tracked_stracks, emb_dists, ious_dists)
            
            dists = np.minimum(ious_dists, statistics)
            #dists = 0.65*statistics + 0.35*ious_dists
            #dists = np.minimum(emb_dists, statistics)
            #dists = statistics
            # if len(dists) == 1 or len(dists) == 0 or emb_dists.size == 0:
            #     dists = dists
            # else:
            #     dists = min_max(dists)
        else:
            dists = ious_dists            
                    
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.8)#要不要在这里也价加个姿态呢？
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        index = [r_tracked_stracks[i[0]].track_id for i in matches]
        indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]
    
    
        for (track_idx, detection_idx), oppo in zip(matches,indices):
            self.tracks[oppo].update(asd_second[detection_idx], detection_idx, 0)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        
        for detection_idx in u_detection:
            if asd[detection_idx]['conf'] < self.new_track_thresh:
                continue
            self._initiate_track(asd[detection_idx], detection_idx)
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > track.end_frame - track.start_frame + 60:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        
        track_ids = set(track.track_id for track in joint_stracks(self.tracked_stracks, self.lost_stracks))
        self.tracks = [track for track in self.tracks if track.track_id in track_ids] 
        track2_dict = {track.track_id: track for track in self.tracks}
        self.tracks = [track2_dict[track.track_id] for track in joint_stracks(self.tracked_stracks, self.lost_stracks) if track.track_id in track2_dict]

        pose_features,targets = [], []
        pose_features_all = []
        active_targets = [t.track_id for t in joint_stracks(self.tracked_stracks,self.lost_stracks)]
        for track in self.tracks:
            pose = [] 

            pose_features += [track.track_data['history'][-1]['pose']]  
            for i in track.track_data['history']:
                pose.append(i['pose'])
            pose_features_all.append(pose)                    
            targets       += [track.track_id]      
        self.metric.partial_fit(np.asarray(pose_features_all), np.asarray(targets), active_targets)
        return output_stracks
        

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.05)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
