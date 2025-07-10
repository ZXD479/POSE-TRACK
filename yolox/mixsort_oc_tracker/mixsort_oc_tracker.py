"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .mixformer import MixFormer
from .association import *
from typing import List, Union, Tuple

from pose_pre import linear_assignment
from pose_pre.track import Track 
from sklearn.linear_model import Ridge
import scipy.stats as stats
import torch
from pose_pre import nn_matching
def linear_assignment1(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, template=None):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        #self.last_observation = bbox
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.template=template

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "eiou": eiou_distance,
                "ct_dist": ct_dist}
class MIXTracker_eiou(object):
    def __init__(self, det_thresh, args, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="eiou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

        self.mixformer=MixFormer(args)
        self.mix_iou=args.mix_iou
        self.alpha=args.alpha

    def re_init(self):
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        #self.det_thresh=det_thresh
        #self.max_age=max_age
        KalmanBoxTracker.count = 0

    def uncovered_area(self, bbox:np.ndarray):
        result=[]
        for i, (x1,y1,x2,y2) in enumerate(bbox):
            w,h=x2-x1,y2-y1
            temp=np.ones((h,w),dtype=np.int8)
            for j, (_x1,_y1,_x2,_y2) in enumerate(bbox):
                if i==j:
                    continue
                _x1-=x1
                _y1-=y1
                _x2-=x1
                _y2-=y1
                if _x1>w or _x2<0 or _y1>h or _y2<0:
                    continue
                temp[max(0,_y1):min(h,_y2),max(0,_x1):min(w,_x2)]=0
            result.append(1-int(temp.sum())/(w*h))
        return result

    def update(self, output_results, img_info, img_size, img, scale=True):

        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        if scale:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        # compute all ious
        max_iou = np.expand_dims(self.uncovered_area(bboxes.astype(np.int32)), axis=-1)

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1),max_iou), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        templates=[x.template for x in self.trackers]
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, t]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # discard invalid track predictions
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate_eiou(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, img, self.mixformer,self.alpha,templates)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :-1])
            det=np.array(dets[m[0]])
            det[2:4]=det[2:4]-det[0:2]
            if det[-1]<self.mix_iou:
                self.trackers[m[1]].template=self.mixformer.crop_and_resize(img,det,"template")

        """
            Second round of associaton by OCR
        """

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets[:,:4], left_trks[:,:4], 0.5)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                vit=self.mixformer.compute_vit_sim(left_dets,left_trks,img,templates)
                rematched_indices = linear_assignment1(-((self.alpha*iou_left)+(1-self.alpha)*vit.T))
                #rematched_indices = linear_assignment1(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :-1])
                    det=np.array(dets[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            det=np.array(dets[i])
            det[2:4]=det[2:4]-det[0:2]
            trk = KalmanBoxTracker(dets[i, :-1], delta_t=self.delta_t,template=self.mixformer.crop_and_resize(img,det,"template"))
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment1(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))
class MIXTracker_reid(object):
    def __init__(self, det_thresh, args, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

        self.mixformer=MixFormer(args)
        self.mix_iou=args.mix_iou
        self.alpha=args.alpha

    def re_init(self):
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        #self.det_thresh=det_thresh
        #self.max_age=max_age
        KalmanBoxTracker.count = 0

    def uncovered_area(self, bbox:np.ndarray):
        result=[]
        for i, (x1,y1,x2,y2) in enumerate(bbox):
            w,h=x2-x1,y2-y1
            temp=np.ones((h,w),dtype=np.int8)
            for j, (_x1,_y1,_x2,_y2) in enumerate(bbox):
                if i==j:
                    continue
                _x1-=x1
                _y1-=y1
                _x2-=x1
                _y2-=y1
                if _x1>w or _x2<0 or _y1>h or _y2<0:
                    continue
                temp[max(0,_y1):min(h,_y2),max(0,_x1):min(w,_x2)]=0
            result.append(1-int(temp.sum())/(w*h))
        return result

    def update(self, output_results, img_info, img_size, img, embedding, scale=True):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        if scale:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        # compute all ious
        max_iou = np.expand_dims(self.uncovered_area(bboxes.astype(np.int32)), axis=-1)

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1),max_iou), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        if self.args.with_reid:
            embedding = embedding[inds_low]
            features_keep = embedding[remain_inds]
        # get predicted locations from existing trackers.
        templates=[x.template for x in self.trackers]
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, t]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # discard invalid track predictions
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, img, self.mixformer,self.alpha,templates)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :-1])
            det=np.array(dets[m[0]])
            det[2:4]=det[2:4]-det[0:2]
            if det[-1]<self.mix_iou:
                self.trackers[m[1]].template=self.mixformer.crop_and_resize(img,det,"template")

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                vit=self.mixformer.compute_vit_sim(dets_second,u_trks,img,templates)
                matched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*vit.T))
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :-1])
                    to_remove_trk_indices.append(trk_ind)
                    det=np.array(dets_second[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                vit=self.mixformer.compute_vit_sim(left_dets,left_trks,img,templates)
                rematched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*vit.T))
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :-1])
                    det=np.array(dets[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            det=np.array(dets[i])
            det[2:4]=det[2:4]-det[0:2]
            trk = KalmanBoxTracker(dets[i, :-1], delta_t=self.delta_t,template=self.mixformer.crop_and_resize(img,det,"template"))
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment1(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))

class MIXTracker(object):
    def __init__(self, det_thresh, args, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

        self.mixformer=MixFormer(args)
        self.mix_iou=args.mix_iou
        self.alpha=args.alpha

    def re_init(self):
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        #self.det_thresh=det_thresh
        #self.max_age=max_age
        KalmanBoxTracker.count = 0

    def uncovered_area(self, bbox:np.ndarray):
        result=[]
        for i, (x1,y1,x2,y2) in enumerate(bbox):
            w,h=x2-x1,y2-y1
            temp=np.ones((h,w),dtype=np.int8)
            for j, (_x1,_y1,_x2,_y2) in enumerate(bbox):
                if i==j:
                    continue
                _x1-=x1
                _y1-=y1
                _x2-=x1
                _y2-=y1
                if _x1>w or _x2<0 or _y1>h or _y2<0:
                    continue
                temp[max(0,_y1):min(h,_y2),max(0,_x1):min(w,_x2)]=0
            result.append(1-int(temp.sum())/(w*h))
        return result

    def update(self, output_results, img_info, img_size, img, scale=True):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        if scale:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        # compute all ious
        max_iou = np.expand_dims(self.uncovered_area(bboxes.astype(np.int32)), axis=-1)

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1),max_iou), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        templates=[x.template for x in self.trackers]
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, t]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # discard invalid track predictions
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, img, self.mixformer,self.alpha,templates)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :-1])
            det=np.array(dets[m[0]])
            det[2:4]=det[2:4]-det[0:2]
            if det[-1]<self.mix_iou:
                self.trackers[m[1]].template=self.mixformer.crop_and_resize(img,det,"template")

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                vit=self.mixformer.compute_vit_sim(dets_second,u_trks,img,templates)
                matched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*vit.T))
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :-1])
                    to_remove_trk_indices.append(trk_ind)
                    det=np.array(dets_second[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                vit=self.mixformer.compute_vit_sim(left_dets,left_trks,img,templates)
                rematched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*vit.T))
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :-1])
                    det=np.array(dets[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            det=np.array(dets[i])
            det[2:4]=det[2:4]-det[0:2]
            trk = KalmanBoxTracker(dets[i, :-1], delta_t=self.delta_t,template=self.mixformer.crop_and_resize(img,det,"template"))
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > trk.age + self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment1(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))

class MIXTracker_pose(object):
    def __init__(self, cfg, det_thresh, args, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="eiou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """ 
        self.cfg = cfg

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.mixformer=MixFormer(args)
        self.mix_iou=args.mix_iou
        self.alpha=args.alpha
        self.tracks = []
        self.n_init = 5
        self.A_dim = 4096
        self.P_dim = 4096
        self.L_dim = 99
        self.mode = ''


    def re_init(self):
        self.trackers = [] #type: List[KalmanBoxTracker]
        self.frame_count = 0
        #self.det_thresh=det_thresh
        #self.max_age=max_age
        self._next_id = 0
        KalmanBoxTracker.count = 0
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.tracks = []

    def uncovered_area(self, bbox:np.ndarray):
        result=[]
        for i, (x1,y1,x2,y2) in enumerate(bbox):
            w,h=x2-x1,y2-y1
            temp=np.ones((h,w),dtype=np.int8)
            for j, (_x1,_y1,_x2,_y2) in enumerate(bbox):
                if i==j:
                    continue
                _x1-=x1
                _y1-=y1
                _x2-=x1
                _y2-=y1
                if _x1>w or _x2<0 or _y1>h or _y2<0:
                    continue
                temp[max(0,_y1):min(h,_y2),max(0,_x1):min(w,_x2)]=0
            result.append(1-int(temp.sum())/(w*h))
        return result
    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1
    def get_prediction_interval(self, y, y_hat, x, x_hat):
        n     = y.size
        resid = y - y_hat
        s_err = np.sqrt(np.sum(resid**2) / (n-2))                    # standard deviation of the error
        t     = stats.t.ppf(0.975, n - 2)                            # used for CI and PI bands
        pi    = t * s_err * np.sqrt( 1 + 1/n + (x_hat - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        return pi

    def forward_for_tracking(self, vectors, attibute="A", time=1):
        
        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]
        
            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, pose_dim)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            with torch.no_grad():
                pose_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)
            
            return pose_pred.cpu()


        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                y0_                  = np.array(en_loca_xy[bs, :, 44, 1])
                n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_                   = np.array(en_time[bs, :])

                loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                if(self.cfg.phalp.distance_type=="EQ_020" or self.cfg.phalp.distance_type=="EQ_021"):
                    loc_                 = 1
                else:
                    loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = self.get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = self.get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = self.get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt    
    def accumulate_vectors(self, track_ids, features="APL"):
        
        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.cfg.phalp.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                #P是pose L是位置
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.cfg.phalp.track_history)]))
            is_tracks = 1

        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        if(is_tracks):#预测位置和姿态
            with torch.no_grad():
                if("P" in features): p_pred = self.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None, 
                                                     loca=l_pred[p_id] if("L" in features) else None)
                
        
    def _match(self, detections, tracked_stracks, appce, iou_dists, detection_indices=None):

        def gated_metric(tracks, dets, track_indices, detection_indices, appce, iou_dists):

            #loca_emb          = np.array([dets[i]['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i]['pose'] for i in detection_indices])
            appce             = appce
            iou_dists         = iou_dists
            targets           = np.array([tracked_stracks[i].id for i in track_indices])
            end_time           = np.array([tracked_stracks[i].time_since_update for i in track_indices])

            cost_matrix       = self.metric.distance([pose_emb, appce, iou_dists.T], targets, end_time, self.frame_count)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # confirmed_tracks = []
        # #confirmed_tracks = [i for i, t in enumerate(self.tracked_stracks) if t.is_activated() ]
        # for i, t in enumerate(self.tracked_stracks):
        #     if t.is_activated():
        #         confirmed_tracks.append(i)

        # Associate confirmed tracks using appearance features.
        #matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        #matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, joint_stracks(self.tracked_stracks, self.lost_stracks), detections, confirmed_tracks)
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, tracked_stracks, detections, appce,  iou_dists, detection_indices = detection_indices)
        
        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d['ground_truth'] for i, d in enumerate(detections)]


        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        #return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix]
    def update(self, asd, output_results, img_info, img_size, img, flag, frame_id, scale=True):
        
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        if scale:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        # compute all ious
        max_iou = np.expand_dims(self.uncovered_area(bboxes.astype(np.int32)), axis=-1)

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1),max_iou), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        asd = [item for item, i in zip(asd, remain_inds) if i == True]
        #asd = asd[remain_inds]
        # get predicted locations from existing trackers.
        templates=[x.template for x in self.trackers]
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, t]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # discard invalid track predictions
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        # matches1, unmatched_tracks, unmatched_detections, statistics = self._match(asd, self.trackers)
        # cost_pose_matrix = statistics[0]
        # #cost_pose_matrix = []
        # matched, unmatched_dets, unmatched_trks = associate_pose(
        #     dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, img, self.mixformer,self.alpha,templates, cost_pose_matrix)
        
        #matches1, unmatched_tracks, unmatched_detections, statistics = self._match(asd, self.trackers)
        #cost_pose_matrix = statistics[0]
        matched, unmatched_dets, unmatched_trks, cost_matrix = associate_pose_reid(#这里没错，就不需要预测的track
            asd, self.trackers, self._match, dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, img, self.mixformer,self.alpha,templates)
        


        index = [self.trackers[i[1]].id for i in matched]
        u_index = [self.trackers[i].id for i in unmatched_trks]
        indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]
        u_indices = [i for i, track in enumerate(self.tracks) if track.track_id in u_index]
                    
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :-1])
            det=np.array(dets[m[0]])
            det[2:4]=det[2:4]-det[0:2]
            if det[-1]<self.mix_iou:
                self.trackers[m[1]].template=self.mixformer.crop_and_resize(img,det,"template")

        
        for (detection_idx,track_idx), oppo in zip(matched,indices):
                self.tracks[track_idx].update(asd[detection_idx], detection_idx, 0)#这里不对，二次匹配的时候带有预测不应该是这样
            
        """
            Second round of associaton by OCR
        """

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_1 = [asd[i] for i in unmatched_dets]
            #extracted_submatrix = np.array([cost_matrix[row, unmatched_trks] for row in unmatched_dets])
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            left_trks_1 = [self.trackers[i] for i in unmatched_trks]
            #iou_left = self.asso_func(left_dets[:,:4], left_trks[:,:4], expand=0.5)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                #第一个是det第二个是track
                vit=self.mixformer.compute_vit_sim(left_dets,left_trks,img,templates)
                #_,_,_,sff = self._match(left_dets_1, left_trks_1, vit, unmatched_dets)
                rematched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*vit.T))
                #rematched_indices = linear_assignment1(-(self.alpha*iou_left+(1-self.alpha)*sff[0].T))
                
                index = [self.trackers[i[1]].id for i in rematched_indices]     
                indices = [i for i, track in enumerate(self.tracks) if track.track_id in index]

                

                # for (detection_idx, track_idx), oppo in zip(rematched_indices,indices):#77.5应该也没这个东西
                #     if iou_left[detection_idx, track_idx] < self.iou_threshold:
                #         continue
                #     self.tracks[oppo].update(asd[detection_idx], detection_idx, 0)#这里不对，二次匹配的时候带有预测不应该是这样
                        
                
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :-1])
                    self.tracks[trk_ind].update(asd[det_ind], det_ind, 0)
                    det=np.array(dets[det_ind])
                    det[2:4]=det[2:4]-det[0:2]
                    if det[-1]<self.mix_iou:
                        self.trackers[trk_ind].template=self.mixformer.crop_and_resize(img,det,"template")
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
#常看
    
        for m in unmatched_trks:
            self.trackers[m].update(None)
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            det=np.array(dets[i])
            det[2:4]=det[2:4]-det[0:2]
            trk = KalmanBoxTracker(dets[i, :-1], delta_t=self.delta_t,template=self.mixformer.crop_and_resize(img,det,"template"))
            self.trackers.append(trk)
        i = len(self.trackers)

        for detection_idx in unmatched_dets:
            self._initiate_track(asd[detection_idx], detection_idx)

        track_ids = set(track.id for track in self.trackers)
        self.tracks = [track for track in self.tracks if track.track_id in track_ids]        
        track2_dict = {track.track_id: track for track in self.tracks}
        self.tracks = [track2_dict[track.id] for track in self.trackers if track.id in track2_dict]

        #self.tracks = [t for t in self.tracks if not t.is_deleted()]
        loca_features, pose_features,targets = [], [], []
        active_targets = [t.id for t in self.trackers]
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
            #loca_features += [track.track_data['history'][-1]['loca']]#而且history只会存储6个，不会存多了
            pose_features += [track.track_data['history'][-1]['pose']]
            targets       += [track.track_id]
            
            
        #self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        self.metric.partial_fit(np.asarray(pose_features), np.asarray(targets), active_targets)
        #return matches
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            # if trk.time_since_update != 0:
            #     print('aa')
            #if(trk.time_since_update > trk.age + self.max_age):
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                #self.tracks.pop(i)
        track_ids = set(track.id for track in self.trackers)
        self.tracks = [track for track in self.tracks if track.track_id in track_ids]        
        track2_dict = {track.track_id: track for track in self.tracks}
        self.tracks = [track2_dict[track.id] for track in self.trackers if track.id in track2_dict]

        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
        
    
    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))

