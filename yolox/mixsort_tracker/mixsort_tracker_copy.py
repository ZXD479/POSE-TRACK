import sys  
sys.path.append('MixViT')
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
import cv2
import copy
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
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

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
        ret = self.last_tlwh.copy()
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
class STrack_eiou(BaseTrack):
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
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

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
        ret = self.last_tlwh.copy()
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

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, iou=0):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.template = None
        self._iou = iou
        self.last_tlwh = self._tlwh
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, template):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.template = template

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, template=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.last_tlwh = new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        if template is not None:
            self.template = template

    def update(self, new_track, frame_id, template=None):
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
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )

        self.last_tlwh = new_tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if template is not None:
            self.template = template

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def last_tlbr(self):
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)
    

class STrack_eiou1(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=30):

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
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

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
        ret = self.last_tlwh.copy()
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

class MIXTracker_pose(object):
    def __init__(self, cfg, args, pose_predictor, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.cfg = cfg
        self.frame_id = 0
        self.args = args
        self.max_age          = 30
        self.A_dim = 4096
        self.P_dim = 4096
        self.L_dim = 99
        self.n_init           = 5
        self.pose_predictor = pose_predictor
        self._next_id         = 1
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.tracks           = []
        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh#100和1
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
        self.t = 0
        prj_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../MixViT")
        )
        self.settings.cfg_file = os.path.join(
            prj_dir, f"experiments/{args.script}/{args.config}.yaml"
        )
        config_module = importlib.import_module(
            "MixViT.lib.config.%s.config" % self.settings.script_name
        )
        self.cfg1 = config_module.cfg
        config_module.update_config_from_file(self.settings.cfg_file)
        update_settings(self.settings, self.cfg1)

        # need modification, for distributed
        network = build_mixformer_deit(self.cfg1)
        self.network = network.cuda(torch.device(f"cuda:{args.local_rank}"))
        self.network.eval()

    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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
        

        self.metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)

    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1
    def visualize(self, logger: SummaryWriter, template, search, search_box):
        # utils for debugging
        logger.add_image("template", template)
        for box in search_box:
            box[2:4] = box[0:2] + box[2:4]
        logger.add_image_with_boxes("search", search, search_box)

    def visualize_box(self, logger: SummaryWriter, img, dets:List[STrack], name):
        # utils for debugging
        logger.add_image_with_boxes(name, img, np.array([s.tlbr for s in dets]),labels=[str(i) for i in range(len(dets))])

    def crop_and_resize(
        self, img: torch.Tensor, center: np.ndarray, s: str, annos: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """crop&resize the `img` centered at `center` and transform `annos` to the cropped position.

        Args:
            img (torch.Tensor): image to be cropped
            center (np.ndarray): center coord
            s (str): 'template' or 'search'
            annos (torch.Tensor, optional): boxes to be transformed. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor,torch.Tensor],torch.Tensor]: transfromed image (and boxes)
        """
        # compute params
        center = torch.from_numpy(center.astype(np.int))
        search_area_factor = self.settings.search_area_factor[s]
        output_sz = self.settings.output_sz[s]
        x, y, w, h = [int(i) for i in center]
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # x:left, y:top
        x = int(round(x + 0.5 * w - crop_sz * 0.5))
        y = int(round(y + 0.5 * h - crop_sz * 0.5))

        try:
            resized_img = resized_crop(
                img, y, x, crop_sz, crop_sz, [output_sz, output_sz]
            )
        except:  # too small box
            zero_img = torch.zeros((3, output_sz, output_sz)).cuda()
            return zero_img if annos is None else zero_img, []

        if annos is not None:
            # (origin_x - x, origin_y - y, origin_w, origin_h)/factor
            transforemd_coord = torch.cat(
                (annos[:, 0:2] - torch.tensor([x, y]), annos[:, 2:4]), dim=1
            )
            return resized_img, transforemd_coord / (crop_sz / output_sz)
        else:
            return resized_img
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
                
        
    def _match(self, detections, tracked_stracks):

        def gated_metric(tracks, dets, track_indices, detection_indices):

            loca_emb          = np.array([dets[i]['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i]['pose'] for i in detection_indices])

            targets           = np.array([joint_stracks(tracked_stracks, self.lost_stracks)[i].track_id for i in track_indices])
            
            cost_matrix       = self.metric.distance([loca_emb, pose_emb], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=None)
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
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, joint_stracks(tracked_stracks, self.lost_stracks), detections)
        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        #if(self.cfg.use_gt): 
        if(False): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    @torch.no_grad()
    def compute_mix_dist(
        self,
        stracks: List[STrack],
        dets: List[STrack],
        img: torch.Tensor,
        fuse: bool = False,
    ) -> np.ndarray:

        # Predict the current location with KF
        STrack.multi_predict(stracks)
        # compute iou dist
        iou = matching.iou_distance(stracks, dets)
        if fuse:
            iou = matching.fuse_score(iou, dets)

        if len(stracks) * len(dets) == 0:
            return iou

        # for every strack, compute its vit-dist with dets
        search_bbox = torch.stack(
            [torch.from_numpy(det.tlwh.astype(np.int)) for det in dets]
        )
        search_imgs = []
        search_boxes = []
        # vit dist
        # self.logger=SummaryWriter('./debug_tensorboard')
        # self.visualize(self.logger,template_imgs[0],search_img,search_box.clone())
        # self.visualize_box(self.logger,img,stracks,"stracks")
        # self.visualize_box(self.logger,img,dets,"dets")
        vit = np.zeros((len(stracks), len(dets)), dtype=np.float64)
        template_imgs = [s.template for s in stracks]
        template_imgs = [img[0] for img in template_imgs if isinstance(img[0], torch.Tensor)]#
        for strack in stracks:
            # centered at predicted position
            center = strack.tlwh
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, center, "search", search_bbox)
            search_imgs.append(s_img)
            search_boxes.append(s_box)

        # img transform & compute
        template_imgs = normalize(
            torch.stack(template_imgs).float().div(255),
            self.cfg1.DATA.MEAN,
            self.cfg1.DATA.STD,
        )
        search_imgs = normalize(
            torch.stack(search_imgs).float().div(255),
            self.cfg1.DATA.MEAN,
            self.cfg1.DATA.STD,
        )
        heatmap = self.network(template_imgs, search_imgs).cpu().detach().numpy()#热力图
        # linear transform to [0,1]
        for i in range(heatmap.shape[0]):
            heatmap[i][0] = heatmap[i][0] - heatmap[i][0].min()
            heatmap[i][0] = heatmap[i][0] / heatmap[i][0].max()

        # compute similarity
        search_size = s_img[0].shape[-1]
        heatmap_size = heatmap.shape[-1]
        factor = search_size // heatmap_size
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_size and cy < search_size:
                    cx, cy = int(cx) // factor, int(cy) // factor
                    top = max(0, cy - self.radius)
                    bottom = min(heatmap_size, cy + self.radius + 1)
                    left = max(0, cx - self.radius)
                    right = min(heatmap_size, cx + self.radius + 1)
                    vit[i][j] = heatmap[i][0][top:bottom, left:right].mean()
                    # vit[i][j] = heatmap[i][0][cy][cx]

        # fuse iou&vit cost
        return self.alpha * iou + (1 - self.alpha) * (1 - vit)
        # if iou.min()<self.args.fuse_iou_thresh:
        #     return iou
        # else:
        #     return 1-vit
        # vit=1-vit
        # for i in range(iou.shape[0]):
        #     for j in range(iou.shape[1]):
        #         if iou[i][j]>self.args.fuse_iou_thresh and vit[i][j]<self.args.fuse_vit_thresh:
        #             iou[i][j]=vit[i][j]
        # return iou
    def z_score_normalize(self, matrix):  
        mean = np.mean(matrix, axis=0)  # 计算每列的平均值  
        std = np.std(matrix, axis=0)    # 计算每列的标准差  
        normalized_matrix = (matrix - mean) / std  # 进行Z-score标准化  
        return normalized_matrix  
    def min_max(self,array):
        if array.size > 0:  
            # 计算Min-Max归一化所需的最小值和最大值  
            min_val = np.min(array)  
            max_val = np.max(array)  
            
            # 应用Min-Max归一化公式  
            normalized_array = (array - min_val) / (max_val - min_val)  
        return normalized_array
    def update(self, asd, output_results, img_info, img_size, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # compute all ious
        all_dets = [x for x in bboxes]
        iou = matching.ious(all_dets, all_dets)
        # compute max iou for every det
        max_iou = []
        for i in range(len(all_dets)):
            iou[i][i] = 0
            max_iou.append(iou[i].max())
        max_iou = np.array(max_iou)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        max_iou_keep = max_iou[remain_inds]
        max_iou_second = max_iou[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        print(self.frame_id)

        asd = [item for item in asd if item['conf'].item() >= self.args.track_thresh]  
        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets, scores_keep, max_iou_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        confirmed_tracks = []
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for i, track in enumerate(self.tracked_stracks):
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                confirmed_tracks.append(i)

        """ Step 2: First association, with high score detection boxes"""

        matches1, unmatched_tracks, unmatched_detections, statistics = self._match(asd, tracked_stracks)
        cost_pose_matrix = statistics[0]#,代价矩阵第一个是track，第二个是detection
        # if self.lost_stracks:
        #     print('asd')
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)


        dists = self.compute_mix_dist(strack_pool, detections, img, fuse=True)#这个就是cost_matrix
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        if dists.size > 0:          
            #dists = 0.25*self.min_max(cost_pose_matrix) + dists
            dists = self.min_max(cost_pose_matrix)
            #dists = self.z_score_normalize(cost_pose_matrix) + self.z_score_normalize(dists)#求和
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

                
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(asd[detection_idx], detection_idx, 0)
        self.accumulate_vectors([i[0] for i in matches], features=self.cfg.phalp.predict)
 
        for track_idx in u_track:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(u_track, features=self.cfg.phalp.predict)
        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets_second, scores_second, max_iou_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        #就在这
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self.compute_mix_dist(r_tracked_stracks, detections_second, img)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = self.compute_mix_dist(unconfirmed, detections, img, fuse=True)
        # dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            unconfirmed[itracked].update(detections[idet], self.frame_id, template)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""

        for detection_idx in u_detection:
            self._initiate_track(asd[detection_idx], detection_idx)
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # do not consider iou constraint
            track.activate(
                self.kalman_filter,
                self.frame_id,
                self.crop_and_resize(img, track._tlwh, "template"),
            )
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        if self.lost_stracks:
            self.t+=1
            print(self.t)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        self.last_img = img
        loca_features, pose_features,targets = [], [], []
        active_targets = [t.track_id for t in joint_stracks(self.tracked_stracks,self.lost_stracks) if t.is_activated]
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
                         
            loca_features += [track.track_data['history'][-1]['loca']]#而且history只会存储6个，不会存多了
            pose_features += [track.track_data['history'][-1]['pose']]

            # loca_features += [track.track_data['prediction']['loca'][-1]]#而且history只会存储6个，不会存多了
            # pose_features += [track.track_data['prediction']['pose'][-1]]

            targets       += [track.track_id]
            
            
        #self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        self.metric.partial_fit(np.asarray(loca_features), np.asarray(pose_features), np.asarray(targets), active_targets)
        #return matches

        return output_stracks
    
class MIXTracker_eiou(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = 0.6 + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.last_img = None
        self.alpha = args.alpha
        self.radius = args.radius
        self.iou_thresh = args.iou_thresh

        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
        prj_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../MixViT")
        )
        self.settings.cfg_file = os.path.join(
            prj_dir, f"experiments/{args.script}/{args.config}.yaml"
        )
        config_module = importlib.import_module(
            "MixViT.lib.config.%s.config" % self.settings.script_name
        )
        self.cfg = config_module.cfg
        config_module.update_config_from_file(self.settings.cfg_file)
        update_settings(self.settings, self.cfg)

        # need modification, for distributed
        network = build_mixformer_deit(self.cfg)
        self.network = network.cuda(torch.device(f"cuda:{args.local_rank}"))
        self.network.eval()

    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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

    def visualize(self, logger: SummaryWriter, template, search, search_box):
        # utils for debugging
        logger.add_image("template", template)
        for box in search_box:
            box[2:4] = box[0:2] + box[2:4]
        logger.add_image_with_boxes("search", search, search_box)

    def visualize_box(self, logger: SummaryWriter, img, dets:List[STrack], name):
        # utils for debugging
        logger.add_image_with_boxes(name, img, np.array([s.tlbr for s in dets]),labels=[str(i) for i in range(len(dets))])

    def crop_and_resize(
        self, img: torch.Tensor, img_name,  center: np.ndarray, s: str, annos: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # compute params
        center = torch.from_numpy(center.astype(np.int))
        search_area_factor = self.settings.search_area_factor[s]
        output_sz = self.settings.output_sz[s]
        x, y, w, h = [int(i) for i in center]


        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # x:left, y:top
        x = int(round(x + 0.5 * w - crop_sz * 0.5))
        y = int(round(y + 0.5 * h - crop_sz * 0.5))

        # frame = cv2.imread('/data2/zxd/sportsmot_publish/dataset/val/' + img_name)
        # cv2.rectangle(frame, (x, y), (x + crop_sz, y + crop_sz), (255, 0, 0), 2)  
        # cv2.circle(frame, (x+ crop_sz//2, y + crop_sz//2), 5, (0,255,0), -1)
        # cv2.imwrite('output_frames/aaa.jpg',frame)

        try:
            resized_img = resized_crop(
                img, y, x, crop_sz, crop_sz, [output_sz, output_sz]
            )
        except:  # too small box
            zero_img = torch.zeros((3, output_sz, output_sz)).cuda()
            return zero_img if annos is None else zero_img, []

        if annos is not None:
            # (origin_x - x, origin_y - y, origin_w, origin_h)/factor
            transforemd_coord = torch.cat(
                (annos[:, 0:2] - torch.tensor([x, y]), annos[:, 2:4]), dim=1
            )
            return resized_img, transforemd_coord / (crop_sz / output_sz)
        else:
            return resized_img

    @torch.no_grad()
    def compute_mix_dist(
        self,
        stracks: List[STrack],
        img_name, 
        dets: List[STrack],
        img: torch.Tensor,
        cur_expand_scale,
        fuse: bool = False,
    ) -> np.ndarray:
        """compute mix distance between stracks and dets.

        Args:
            stracks (List[STrack]): len = m
            dets (List[STrack]): len = n
            img (torch.Tensor): current image
            fuse (bool, optional): whether to fuse det score into iou. Defaults to False.

        Returns:
            np.ndarray: m x n
        """

        # Predict the current location with KF
        STrack.multi_predict(stracks)
        # compute iou dist
        #iou = matching.iou_distance(stracks, dets)
        iou = matching.eiou_distance(stracks, dets, cur_expand_scale)
        if fuse:
            iou = matching.fuse_score(iou, dets)

        if len(stracks) * len(dets) == 0:
            return iou

        # for every strack, compute its vit-dist with dets
        search_bbox = torch.stack(
            [torch.from_numpy(det.tlwh.astype(np.int)) for det in dets]
        )
        search_imgs = []
        search_boxes = []
        # vit dist

        template_imgs = [s.template for s in stracks]
        #template_imgs = [img[0] for img in template_imgs if isinstance(img[0], torch.Tensor)]#
        for strack in stracks:
            # centered at predicted position
            center = strack.tlwh
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, img_name,  center, "search", search_bbox)
            search_imgs.append(s_img)
            search_boxes.append(s_box)
        # self.logger=SummaryWriter('./debug_tensorboard')
        # self.visualize(self.logger,template_imgs[0],search_imgs,search_boxes.clone())
        # self.visualize_box(self.logger,img,stracks,"stracks")
        # self.visualize_box(self.logger,img,dets,"dets")
        vit = np.zeros((len(stracks), len(dets)), dtype=np.float64)
        # img transform & compute
        template_imgs = normalize(
            torch.stack(template_imgs).float().div(255),#96，96
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        search_imgs = normalize(
            torch.stack(search_imgs).float().div(255),
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        heatmap = self.network(template_imgs, search_imgs).cpu().detach().numpy()#热力图
        # linear transform to [0,1]
        for i in range(heatmap.shape[0]):
            heatmap[i][0] = heatmap[i][0] - heatmap[i][0].min()
            heatmap[i][0] = heatmap[i][0] / heatmap[i][0].max()

        # compute similarity
        search_size = s_img[0].shape[-1]
        heatmap_size = heatmap.shape[-1]
        factor = search_size // heatmap_size
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_size and cy < search_size:
                    cx, cy = int(cx) // factor, int(cy) // factor
                    top = max(0, cy - self.radius)
                    bottom = min(heatmap_size, cy + self.radius + 1)
                    left = max(0, cx - self.radius)
                    right = min(heatmap_size, cx + self.radius + 1)
                    vit[i][j] = heatmap[i][0][top:bottom, left:right].mean()
                    # vit[i][j] = heatmap[i][0][cy][cx]

        # fuse iou&vit cost
        #return self.alpha * iou + (1 - self.alpha) * (1 - vit)#高，不对，vit全乱了，试试原始的会不会乱
        return iou 
    def update(self, output_results, embedding):
        
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''
        
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
                

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            
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
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]

            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
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
            #if self.args.with_keypoints:

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
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
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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

        return output_stracks
class MIXTracker_eiou1(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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

        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
        prj_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../MixViT")
        )
        self.settings.cfg_file = os.path.join(
            prj_dir, f"experiments/{args.script}/{args.config}.yaml"
        )
        config_module = importlib.import_module(
            "MixViT.lib.config.%s.config" % self.settings.script_name
        )
        self.cfg = config_module.cfg
        config_module.update_config_from_file(self.settings.cfg_file)
        update_settings(self.settings, self.cfg)

        # need modification, for distributed
        network = build_mixformer_deit(self.cfg)
        self.network = network.cuda(torch.device(f"cuda:{args.local_rank}"))
        self.network.eval()

    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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

    def visualize(self, logger: SummaryWriter, template, search, search_box):
        # utils for debugging
        logger.add_image("template", template)
        for box in search_box:
            box[2:4] = box[0:2] + box[2:4]
        logger.add_image_with_boxes("search", search, search_box)

    def visualize_box(self, logger: SummaryWriter, img, dets:List[STrack], name):
        # utils for debugging
        logger.add_image_with_boxes(name, img, np.array([s.tlbr for s in dets]),labels=[str(i) for i in range(len(dets))])

    def crop_and_resize(
        self, img: torch.Tensor, center: np.ndarray, s: str, annos: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """crop&resize the `img` centered at `center` and transform `annos` to the cropped position.

        Args:
            img (torch.Tensor): image to be cropped
            center (np.ndarray): center coord
            s (str): 'template' or 'search'
            annos (torch.Tensor, optional): boxes to be transformed. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor,torch.Tensor],torch.Tensor]: transfromed image (and boxes)
        """
        # compute params
        center = torch.from_numpy(center.astype(np.int))
        search_area_factor = self.settings.search_area_factor[s]
        output_sz = self.settings.output_sz[s]
        x, y, w, h = [int(i) for i in center]
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # x:left, y:top
        x = int(round(x + 0.5 * w - crop_sz * 0.5))
        y = int(round(y + 0.5 * h - crop_sz * 0.5))

        try:
            resized_img = resized_crop(
                img, y, x, crop_sz, crop_sz, [output_sz, output_sz]
            )
        except:  # too small box
            zero_img = torch.zeros((3, output_sz, output_sz)).cuda()
            return zero_img if annos is None else zero_img, []

        if annos is not None:
            # (origin_x - x, origin_y - y, origin_w, origin_h)/factor
            transforemd_coord = torch.cat(
                (annos[:, 0:2] - torch.tensor([x, y]), annos[:, 2:4]), dim=1
            )
            return resized_img, transforemd_coord / (crop_sz / output_sz)
        else:
            return resized_img

    @torch.no_grad()
    def compute_mix_dist(
        self,
        stracks: List[STrack],
        dets: List[STrack],
        img: torch.Tensor,
        cur_expand_scale,
        fuse: bool = False,
    ) -> np.ndarray:

        # Predict the current location with KF
        STrack.multi_predict(stracks)
        # compute iou dist
        #iou = matching.iou_distance(stracks, dets)deep_copy = copy.deepcopy(original_list) 
        iou = matching.eiou_distance(copy.deepcopy(stracks), copy.deepcopy(dets), cur_expand_scale)
        if fuse:
            iou = matching.fuse_score(iou, dets)

        if len(stracks) * len(dets) == 0:
            return iou

        # for every strack, compute its vit-dist with dets
        search_bbox = torch.stack(
            [torch.from_numpy(det.tlwh.astype(np.int)) for det in dets]
        )
        search_imgs = []
        search_boxes = []

        vit = np.zeros((len(stracks), len(dets)), dtype=np.float64)
        template_imgs = [s.template for s in stracks]
        #template_imgs = [img[0] for img in template_imgs if isinstance(img[0], torch.Tensor)]#
        for strack in stracks:
            # centered at predicted position
            center = strack.tlwh
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, center, "search", search_bbox)
            search_imgs.append(s_img)
            search_boxes.append(s_box)

        # img transform & compute
        template_imgs = normalize(
            torch.stack(template_imgs).float().div(255),#96，96
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        search_imgs = normalize(
            torch.stack(search_imgs).float().div(255),
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        heatmap = self.network(template_imgs, search_imgs).cpu().detach().numpy()#热力图
        # linear transform to [0,1]
        for i in range(heatmap.shape[0]):
            heatmap[i][0] = heatmap[i][0] - heatmap[i][0].min()
            heatmap[i][0] = heatmap[i][0] / heatmap[i][0].max()

        # compute similarity
        search_size = s_img[0].shape[-1]
        heatmap_size = heatmap.shape[-1]
        factor = search_size // heatmap_size
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_size and cy < search_size:
                    cx, cy = int(cx) // factor, int(cy) // factor
                    top = max(0, cy - self.radius)
                    bottom = min(heatmap_size, cy + self.radius + 1)
                    left = max(0, cx - self.radius)
                    right = min(heatmap_size, cx + self.radius + 1)
                    vit[i][j] = heatmap[i][0][top:bottom, left:right].mean()
                    # vit[i][j] = heatmap[i][0][cy][cx]

        # fuse iou&vit cost
        return 0.6 * iou + 0.4 * (1 - vit)
        # if iou.min()<self.args.fuse_iou_thresh:
        #     return iou
        # else: 
        #     return 1-vit
        # vit=1-vit
        # for i in range(iou.shape[0]):
        #     for j in range(iou.shape[1]):
        #         if iou[i][j]>self.args.fuse_iou_thresh and vit[i][j]<self.args.fuse_vit_thresh:
        #             iou[i][j]=vit[i][j]
        # return iou

    def update(self, output_results, embedding):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:

            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2


        # compute all ious
        lowest_inds = scores > 0.1
        bboxes = bboxes[lowest_inds]
        scores = scores[lowest_inds]

        # Find high threshold detections
        remain_inds = scores > 0.6
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        

        embedding = embedding[lowest_inds]
        features_keep = embedding[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack_eiou1(STrack_eiou1.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets, scores_keep, features_keep)
            ]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
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
            ious_dists_mask = (ious_dists > 0.5)

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > 0.25] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.8)

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

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < 0.6
            inds_low = scores > 0.1
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
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
                detections_second = [STrack_eiou1(STrack_eiou1.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack_eiou1(STrack_eiou1.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > 0.5)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > 0.25] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.7:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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

        return output_stracks
class MIXTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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

        # mixformer setting & cfg
        # adapted from lib/train/run_training.py & train_script_mixformer.py
        self.settings = ws_settings.Settings()
        self.settings.script_name = args.script
        self.settings.config_name = args.config
        prj_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../MixViT")
        )
        self.settings.cfg_file = os.path.join(
            prj_dir, f"experiments/{args.script}/{args.config}.yaml"
        )
        config_module = importlib.import_module(
            "MixViT.lib.config.%s.config" % self.settings.script_name
        )
        self.cfg = config_module.cfg
        config_module.update_config_from_file(self.settings.cfg_file)
        update_settings(self.settings, self.cfg)

        # need modification, for distributed
        network = build_mixformer_deit(self.cfg)
        self.network = network.cuda(torch.device(f"cuda:{args.local_rank}"))
        self.network.eval()

    def re_init(self, args, frame_rate=30):
        BaseTrack._count = 0 # set to 0 for new video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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

    def visualize(self, logger: SummaryWriter, template, search, search_box):
        # utils for debugging
        logger.add_image("template", template)
        for box in search_box:
            box[2:4] = box[0:2] + box[2:4]
        logger.add_image_with_boxes("search", search, search_box)

    def visualize_box(self, logger: SummaryWriter, img, dets:List[STrack], name):
        # utils for debugging
        logger.add_image_with_boxes(name, img, np.array([s.tlbr for s in dets]),labels=[str(i) for i in range(len(dets))])

    def crop_and_resize(
        self, img: torch.Tensor, center: np.ndarray, s: str, annos: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """crop&resize the `img` centered at `center` and transform `annos` to the cropped position.

        Args:
            img (torch.Tensor): image to be cropped
            center (np.ndarray): center coord
            s (str): 'template' or 'search'
            annos (torch.Tensor, optional): boxes to be transformed. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor,torch.Tensor],torch.Tensor]: transfromed image (and boxes)
        """
        # compute params
        center = torch.from_numpy(center.astype(np.int))
        search_area_factor = self.settings.search_area_factor[s]
        output_sz = self.settings.output_sz[s]
        x, y, w, h = [int(i) for i in center]
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # x:left, y:top
        x = int(round(x + 0.5 * w - crop_sz * 0.5))
        y = int(round(y + 0.5 * h - crop_sz * 0.5))

        try:
            resized_img = resized_crop(
                img, y, x, crop_sz, crop_sz, [output_sz, output_sz]
            )
        except:  # too small box
            zero_img = torch.zeros((3, output_sz, output_sz)).cuda()
            return zero_img if annos is None else zero_img, []

        if annos is not None:
            # (origin_x - x, origin_y - y, origin_w, origin_h)/factor
            transforemd_coord = torch.cat(
                (annos[:, 0:2] - torch.tensor([x, y]), annos[:, 2:4]), dim=1
            )
            return resized_img, transforemd_coord / (crop_sz / output_sz)
        else:
            return resized_img

    @torch.no_grad()
    def compute_mix_dist(
        self,
        stracks: List[STrack],
        dets: List[STrack],
        img: torch.Tensor,
        fuse: bool = False,
    ) -> np.ndarray:

        # Predict the current location with KF
        STrack.multi_predict(stracks)
        # compute iou dist
        iou = matching.iou_distance(stracks, dets)
        if fuse:
            iou = matching.fuse_score(iou, dets)

        if len(stracks) * len(dets) == 0:
            return iou

        # for every strack, compute its vit-dist with dets
        search_bbox = torch.stack(
            [torch.from_numpy(det.tlwh.astype(np.int)) for det in dets]
        )
        search_imgs = []
        search_boxes = []
        # vit dist
        # self.logger=SummaryWriter('./debug_tensorboard')
        # self.visualize(self.logger,template_imgs[0],search_img,search_box.clone())
        # self.visualize_box(self.logger,img,stracks,"stracks")
        # self.visualize_box(self.logger,img,dets,"dets")
        vit = np.zeros((len(stracks), len(dets)), dtype=np.float64)
        template_imgs = [s.template for s in stracks]
        #template_imgs = [img[0] for img in template_imgs if isinstance(img[0], torch.Tensor)]#
        for strack in stracks:
            # centered at predicted position
            center = strack.tlwh
            # crop search area & transform det coord
            s_img, s_box = self.crop_and_resize(img, center, "search", search_bbox)
            search_imgs.append(s_img)
            search_boxes.append(s_box)

        # img transform & compute
        template_imgs = normalize(
            torch.stack(template_imgs).float().div(255),#96，96
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        search_imgs = normalize(
            torch.stack(search_imgs).float().div(255),
            self.cfg.DATA.MEAN,
            self.cfg.DATA.STD,
        )
        heatmap = self.network(template_imgs, search_imgs).cpu().detach().numpy()#热力图
        # linear transform to [0,1]
        for i in range(heatmap.shape[0]):
            heatmap[i][0] = heatmap[i][0] - heatmap[i][0].min()
            heatmap[i][0] = heatmap[i][0] / heatmap[i][0].max()

        # compute similarity
        search_size = s_img[0].shape[-1]
        heatmap_size = heatmap.shape[-1]
        factor = search_size // heatmap_size
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_size and cy < search_size:
                    cx, cy = int(cx) // factor, int(cy) // factor
                    top = max(0, cy - self.radius)
                    bottom = min(heatmap_size, cy + self.radius + 1)
                    left = max(0, cx - self.radius)
                    right = min(heatmap_size, cx + self.radius + 1)
                    vit[i][j] = heatmap[i][0][top:bottom, left:right].mean()
                    # vit[i][j] = heatmap[i][0][cy][cx]

        # fuse iou&vit cost
        return self.alpha * iou + (1 - self.alpha) * (1 - vit)
        # if iou.min()<self.args.fuse_iou_thresh:
        #     return iou
        # else: 
        #     return 1-vit
        # vit=1-vit
        # for i in range(iou.shape[0]):
        #     for j in range(iou.shape[1]):
        #         if iou[i][j]>self.args.fuse_iou_thresh and vit[i][j]<self.args.fuse_vit_thresh:
        #             iou[i][j]=vit[i][j]
        # return iou

    def update(self, output_results, img_info, img_size, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # compute all ious
        all_dets = [x for x in bboxes]
        iou = matching.ious(all_dets, all_dets)
        # compute max iou for every det
        max_iou = []
        for i in range(len(all_dets)):
            iou[i][i] = 0
            max_iou.append(iou[i].max())
        max_iou = np.array(max_iou)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        max_iou_keep = max_iou[remain_inds]
        max_iou_second = max_iou[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets, scores_keep, max_iou_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists = self.compute_mix_dist(strack_pool, detections, img, fuse=True)#这个就是cost_maxtix
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(#匹配
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, u)
                for (tlbr, s, u) in zip(dets_second, scores_second, max_iou_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self.compute_mix_dist(r_tracked_stracks, detections_second, img)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            template = (self.crop_and_resize(img, det.tlwh, "template") if det._iou < self.iou_thresh else None
            )
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, template)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, template=template)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = self.compute_mix_dist(unconfirmed, detections, img, fuse=True)
        # dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            det = detections[idet]
            template = (
                self.crop_and_resize(img, det.tlwh, "template")
                if det._iou < self.iou_thresh
                else None
            )
            unconfirmed[itracked].update(detections[idet], self.frame_id, template)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # do not consider iou constraint
            track.activate(
                self.kalman_filter,
                self.frame_id,
                self.crop_and_resize(img, track._tlwh, "template"),
            )
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        self.last_img = img
        return output_stracks
class Deep_EIoU_pose(object):
    def __init__(self, cfg, args, pose_predictor=None, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
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
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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


    def _match(self, detections, tracked_stracks, appce):

        def gated_metric(tracks, dets, track_indices, detection_indices, appce):

            loca_emb          = np.array([dets[i]['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i]['pose'] for i in detection_indices])
            appce             = appce
            targets           = np.array([tracked_stracks[i].track_id for i in track_indices])
            
            cost_matrix       = self.metric.distance([loca_emb, pose_emb, appce], targets, dims=[self.A_dim, self.P_dim, self.L_dim],  phalp_tracker=None)
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
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, tracked_stracks , detections, appce)
        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
            
        if appce.size == 0:
            cost_pose_matrix = np.zeros((len(tracked_stracks), len(detections)), dtype=np.float)
        else:         
            if len(cost_matrix) == 1:
               cost_pose_matrix = cost_matrix
            else:
                cost_pose_matrix = matching.min_max(cost_matrix)

        return matches, unmatched_tracks, unmatched_detections, cost_pose_matrix
    def accumulate_vectors(self, track_ids, features="PL"):
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
                
    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1        
    def update(self, asd, output_results, embedding, flag, frame_id):
        
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''
        if flag > 12 or frame_id <= 3:
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
                    

                # Remove bad detections
                lowest_inds = scores > self.track_low_thresh
                bboxes = bboxes[lowest_inds]
                scores = scores[lowest_inds]
                #np.save('bbox144.npy',bboxes)
                # Find high threshold detections
                remain_inds = scores > self.args.track_high_thresh
                dets = bboxes[remain_inds]
                scores_keep = scores[remain_inds]
                asd = [item for item, i in zip(asd, remain_inds) if i == True]
                asd_all = asd
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
            tracked_stracks = []  # type: list[STrack]
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
                #if self.args.with_keypoints:

                if self.args.with_reid:
                    
                    emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    matches1, unmatched_tracks, unmatched_detections, statistics = self._match(asd, strack_pool, emb_dists)

                    dists = np.minimum(ious_dists, statistics)
                    #dists = np.minimum(ious_dists, emb_dists)
                    #dists = 0.8*ious_dists + 0.2 * statistics
                else:
                    dists = ious_dists

                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

                for itracked, idet in matches:
                    track = strack_pool[itracked]
                    det = detections[idet]
                    if track.state == TrackState.Tracked:
                        track.update(detections[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                for track_idx, detection_idx in matches:
                    self.tracks[track_idx].update(asd[detection_idx], detection_idx, 0)
                self.accumulate_vectors([i[0] for i in matches], features=self.mode)
    
                for track_idx in u_track:
                    self.tracks[track_idx].mark_missed()
                self.accumulate_vectors(u_track, features=self.mode)
                
                
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
                    detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s, f) for
                                        (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
                else:
                    detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s) for
                                        (tlbr, s) in zip(dets_second, scores_second)]
            else:
                detections_second = []

            r_tracked_stracks = strack_pool
            dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                matches1, unmatched_tracks, unmatched_detections, statistics = self._match(asd, unconfirmed, emb_dists)
                #dists = 0.8*ious_dists + 0.2 * statistics
                dists = np.minimum(ious_dists, statistics)
                #dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
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
                self._initiate_track(asd_all[detection_idx], detection_idx)
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.new_track_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
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
            
            
            
            
            loca_features, pose_features,targets = [], [], []
            #active_targets = [t.track_id for t in joint_stracks(self.tracked_stracks,self.lost_stracks) if t.is_activated]
            active_targets = [t.track_id for t in joint_stracks(self.tracked_stracks,self.lost_stracks)]
            for track in self.tracks:
                if not (track.is_confirmed() or track.is_tentative()): continue
                        
                            
    
                if self.mode == "PL":       
                    loca_features += [track.track_data['prediction']['loca'][-1]]#而且history只会存储6个，不会存多了
                    pose_features += [track.track_data['prediction']['pose'][-1]]
                else:
                    loca_features += [track.track_data['history'][-1]['loca']]#而且history只会存储6个，不会存多了
                    pose_features += [track.track_data['history'][-1]['pose']]  
                targets       += [track.track_id]
                
                
            #self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
            self.metric.partial_fit(np.asarray(loca_features), np.asarray(pose_features), np.asarray(targets), active_targets)
            #return matches
            return output_stracks
        else:
            
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
                    

                # Remove bad detections
                lowest_inds = scores > self.track_low_thresh
                bboxes = bboxes[lowest_inds]
                scores = scores[lowest_inds]

                # Find high threshold detections
                remain_inds = scores > self.args.track_high_thresh
                dets = bboxes[remain_inds]
                scores_keep = scores[remain_inds]
                
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
                    detections = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s, f) for
                                (tlbr, s, f) in zip(dets, scores_keep, features_keep)]

                else:
                    detections = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s) for
                                (tlbr, s) in zip(dets, scores_keep)]
            else:
                detections = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed = []
            tracked_stracks = []  # type: list[STrack]
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
                #if self.args.with_keypoints:

                if self.args.with_reid:
                    emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(ious_dists, emb_dists)
                else:
                    dists = ious_dists

                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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

            ''' Step 3: Second association, with low score detection boxes'''
            if len(scores):
                inds_high = scores < self.args.track_high_thresh
                inds_low = scores > self.args.track_low_thresh
                inds_second = np.logical_and(inds_low, inds_high)
                dets_second = bboxes[inds_second]
                scores_second = scores[inds_second]
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
                    detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s, f) for
                                        (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
                else:
                    detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s) for
                                        (tlbr, s) in zip(dets_second, scores_second)]
            else:
                detections_second = []

            r_tracked_stracks = strack_pool
            dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.new_track_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
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

            return output_stracks
class Deep_EIoU(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

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

    def update(self, output_results, embedding):
        
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''
        
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
                

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            
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
                detections = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]

            else:
                detections = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
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
            #if self.args.with_keypoints:

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
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
                detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack_eiou(STrack_eiou.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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
