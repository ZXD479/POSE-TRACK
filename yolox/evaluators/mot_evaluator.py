from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch
import cv2
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
# from yolox.tracker.byte_tracker import BYTETracker
#from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
#from yolox.motdt_tracker.motdt_tracker import OnlineTracker#啊啊啊
from pose_pre.base import FullConfig 
from pose_pre.pose_trsanformerv2 import Pose_transformer_v2, smpl_to_pose_camera_vector, get3d_human_features
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
def draw_and_save_results(results, save_folder, frame_id, origin_img):
   
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate through results
    frame_id=results[0]
    online_tlwhs=results[1]
    obj_ids=results[2]
    #online_scores=results[3]
    for i, tlwh in enumerate(online_tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))

        color = (255,0,0)
        cv2.rectangle(origin_img, intbox[0:2], intbox[2:4], color=color, thickness=2)
        cv2.putText(origin_img, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    thickness=2)

        # Save the frame with bounding boxes
    output_file = os.path.join(save_folder, f"{frame_id}.jpg")
    cv2.imwrite(output_file, origin_img)
def draw_and_save_results1(results, save_folder, frame_id, origin_img):
   
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate through results
    frame_id=1
    online_tlwhs=results[:,:4]
    obj_ids=1
    #online_scores=results[3]
    for i, tlwh in enumerate(online_tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, w, h)))
        obj_id = int(i)
        id_text = '{}'.format(int(obj_id))

        color = (255,0,0)
        cv2.rectangle(origin_img, intbox[0:2], intbox[2:4], color=color, thickness=2)
        cv2.putText(origin_img, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    thickness=2)

        # Save the frame with bounding boxes
    output_file = os.path.join(save_folder, f"{frame_id}.jpg")
    cv2.imwrite(output_file, origin_img)
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
         for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(float(score), 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate_byte(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):


        from yolox.byte_tracker.byte_tracker import BYTETracker
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        bboxes = []
        pose_embedding = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1


        cfg = FullConfig()
        model_hmr, model_cfg = load_hmr2(self.args.checkpoint)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_hmr = model_hmr.to(device)
        model_hmr.eval()

            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        np.save('pose_dance_embedding/'+ video_names[video_id - 1] +'.npy', pose_embedding)
                        np.save('pose_dance_bboxes/'+ video_names[video_id - 1] +'.npy', bboxes)
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                        bboxes = []
                        pose_embedding = []
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            
            
            _,_, img_height, img_width = imgs.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]            
            
            scale = min(1440/1920, 800/1080)
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)
            frame = cv2.imread(os.path.join('/data1/zxd/DanceTrack/val/',info_imgs[4][0]))

                      
            if frame_id == 1:
                flag = 0
            asd, flag = get3d_human_features(cfg, model_hmr, outputs[0]/scale, frame, measurments, frame_id, flag)
            bboxes.append(outputs[0].cpu().numpy())
            poses = [d['pose'] for d in asd if 'pose' in d]
            pose_embedding.append(poses)
            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, asd)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)
                np.save('pose_dance_embedding/'+ info_imgs[4][0].split('/')[0] +'.npy', pose_embedding)
                np.save('pose_dance_bboxes/'+ info_imgs[4][0].split('/')[0] +'.npy', bboxes)
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_result
    
    def evaluate_mixsort_oc_pose_soccer(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker,MIXTracker_pose

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []

        data_path = self.args.root_path
        seq_path =  os.path.join(data_path,'detection_soccer/')
        data_folder = 'TrackEval/data/trackers/mot_challenge/soccer-val/data'
        seqs = os.listdir(seq_path)
        seqs = [path.replace('.npy','') for path in seqs if path.endswith('.npy')]
        seqs.sort()
        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        cfg = FullConfig()
        with tqdm(total=36750, desc="Processing files") as pbar:
            for seq in seqs:  
                if not os.path.exists(os.path.join(data_path, 'detection_soccer','{}.npy'.format(seq))):
                    continue

                pose_embedding = np.load(os.path.join(data_path,'pose_embedding_soccer/','{}.npy'.format(seq)),allow_pickle=True)
            
                detections = np.load(os.path.join(data_path, 'detection_soccer','{}.npy'.format(seq)),allow_pickle=True) 
                sct_output_path = os.path.join(data_folder,'{}.txt'.format(seq))
        tracker = MIXTracker_pose(cfg = cfg, det_thresh = self.args.track_thresh,args=self.args, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, max_age=self.args.track_buffer)
        for frame_id,det in enumerate(detections,1):        
            if self.args.val:
                frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/val/',info_imgs[4][0]))
            else:
                frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/test/',info_imgs[4][0]))
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if frame_id == 1:
                flag = 0
            #draw_and_save_results1(outputs[0]/scale, save_folder, frame_id, frame)
            asd, flag = get3d_human_features(cfg, model_hmr, det, frame, [720, 1280, 1280, 0, 280], frame_id, flag, pose_embedding[frame_id-1])
            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(asd, outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda(), flag, frame_id)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            #draw_and_save_results(results[-1], save_folder, frame_id, frame)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_oc_pose(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker,MIXTracker_pose

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        cfg = FullConfig()
        model_hmr, model_cfg = load_hmr2(self.args.checkpoint)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_hmr = model_hmr.to(device)
        model_hmr.eval()

        pose_predictor = Pose_transformer_v2(cfg, model_hmr)
        pose_predictor.load_weights(cfg.pose_predictor.weights_path)
        pose_predictor.to('cuda')
        save_folder = './output_frames1/'    
        alreadly = ['v_9p0i81kAEwE_c009',]
        tracker = MIXTracker_pose(cfg = cfg, pose_predictor= pose_predictor, det_thresh = self.args.track_thresh,args=self.args, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, max_age=self.args.track_buffer)
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            # if info_imgs[4][0].split('/')[0] != 'v_9p0i81kAEwE_c009':
            #     continue
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if frame_id == 1:
                    tracker.re_init()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                        write_results_no_score(result_filename, results)
                        results = []
                # init tracker
                video_id = info_imgs[3].item()
                img_name = img_file_name[0].split("/")[2]
                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized 
                    stack of parameters on all datasets.
                """
                if video_name not in video_names:
                    video_names[video_id] = video_name
                
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if self.args.val:
                frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/val/',info_imgs[4][0]))
            else:
                frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/test/',info_imgs[4][0]))
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)
            _,_, img_height, img_width = origin_imgs.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            scale = min(1440/1280, 800/720)
            if frame_id == 1:
                flag = 0

            asd, flag = get3d_human_features(cfg, model_hmr, outputs[0]/scale, frame, measurments, frame_id, flag)
            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(asd, outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda(), flag, frame_id)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))


            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    
    def evaluate_mixsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        from yolox.mixsort_tracker.mixsort_tracker import MIXTracker

        # like ByteTrack, we use different setting for different videos
        setting = {
                    'MOT17-01-FRCNN':{
                        'track_buffer':27,
                        'track_thresh':0.6275
                    },
                    'MOT17-03-FRCNN':{
                        'track_buffer':31,
                        'track_thresh':0.5722
                    },
                    'MOT17-06-FRCNN':{
                        'track_buffer':16,
                        'track_thresh':0.5446
                    },
                    'MOT17-07-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.5939
                    },
                    'MOT17-08-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.7449
                    },
                    'MOT17-12-FRCNN':{
                        'track_buffer':29,
                        'track_thresh':0.7036
                    },
                    'MOT17-14-FRCNN':{
                        'track_buffer':28,
                        'track_thresh':0.5436
                    },
                }
        
        def set_args(args, video):
            if video not in setting.keys():
                return args
            for k,v in setting[video].items():
                args.__setattr__(k,v)
            return args
        
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

            
        tracker = MIXTracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                try:
                    frame_id = info_imgs[2].item()
                    video_id = info_imgs[3].item()
                    img_file_name = info_imgs[4]
                    video_name = img_file_name[0].split('/')[0]
                    
                    if video_name not in video_names:
                        video_names[video_id] = video_name
                    if frame_id == 1:
                        self.args = set_args(self.args, video_name)
                        if 'MOT17' in video_name:
                            self.args.alpha = 0.8778
                            self.args.iou_thresh = 0.2217
                            self.args.match_thresh = 0.7986
                        tracker.re_init(self.args)
                        if len(results) != 0:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                            write_results(result_filename, results)
                            results = []
                except:
                    frame_id = info_imgs[2][0].item()
                    video_id = info_imgs[3][0].item()
                    img_file_name = info_imgs[4][0]
                    video_name = img_file_name[0].split('/')[0]
                    
                    if video_name not in video_names:
                        video_names[video_id] = video_name
                    if frame_id == 1:
                        self.args = set_args(self.args, video_name)
                        if 'MOT17' in video_name:
                            self.args.alpha = 0.8778
                            self.args.iou_thresh = 0.2217
                            self.args.match_thresh = 0.7986
                        tracker.re_init(self.args)
                        if len(results) != 0:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                            write_results(result_filename, results)
                            results = []
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())#标记
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_eiou(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        if self.args.iou_only:
            from yolox.mixsort_tracker.mixsort_iou_tracker import MIXTracker
        else:
            from yolox.mixsort_tracker.mixsort_tracker import MIXTracker, MIXTracker_eiou

        # like ByteTrack, we use different setting for different videos
        setting = {
                    'MOT17-01-FRCNN':{
                        'track_buffer':27,
                        'track_thresh':0.6275
                    },
                    'MOT17-03-FRCNN':{
                        'track_buffer':31,
                        'track_thresh':0.5722
                    },
                    'MOT17-06-FRCNN':{
                        'track_buffer':16,
                        'track_thresh':0.5446
                    },
                    'MOT17-07-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.5939
                    },
                    'MOT17-08-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.7449
                    },
                    'MOT17-12-FRCNN':{
                        'track_buffer':29,
                        'track_thresh':0.7036
                    },
                    'MOT17-14-FRCNN':{
                        'track_buffer':28,
                        'track_thresh':0.5436
                    },
                }
        
        def set_args(args, video):
            if video not in setting.keys():
                return args
            for k,v in setting[video].items():
                args.__setattr__(k,v)
            return args
        
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        tracker = MIXTracker_eiou(self.args)
        ori_thresh = self.args.track_thresh
        save_folder = './output_frames/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):#这里就已经是1440，800
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    self.args = set_args(self.args, video_name)
                    if 'MOT17' in video_name:
                        self.args.alpha = 0.8778
                        self.args.iou_thresh = 0.2217
                        self.args.match_thresh = 0.7986
                    tracker.re_init(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)#在这里就是1440，800
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                det = outputs[0].cpu().detach().numpy()
                scale = min(1440/info_imgs[1], 800/info_imgs[0])
                det /= scale
                # rows_to_remove = np.any(det[:, 0:4] < 1, axis=1) # remove edge detection
                # det = det[~rows_to_remove]
                online_targets = tracker.update(det, origin_imgs.squeeze(0).cuda(), info_imgs[4][0])#标记
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    #vertical = tlwh[2] / tlwh[3] > 1.6
                    #if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    if tlwh[2] * tlwh[3] > 0.6:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            #frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/val/',info_imgs[4][0]))

            #draw_and_save_results(results[-1], save_folder, frame_id, frame)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_pose(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        if self.args.iou_only:
            from yolox.mixsort_tracker.mixsort_iou_tracker import MIXTracker
        else:
            from yolox.mixsort_tracker.mixsort_tracker import MIXTracker, MIXTracker_pose

        # like ByteTrack, we use different setting for different videos
        setting = {
                    'MOT17-01-FRCNN':{
                        'track_buffer':27,
                        'track_thresh':0.6275
                    },
                    'MOT17-03-FRCNN':{
                        'track_buffer':31,
                        'track_thresh':0.5722
                    },
                    'MOT17-06-FRCNN':{
                        'track_buffer':16,
                        'track_thresh':0.5446
                    },
                    'MOT17-07-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.5939
                    },
                    'MOT17-08-FRCNN':{
                        'track_buffer':24,
                        'track_thresh':0.7449
                    },
                    'MOT17-12-FRCNN':{
                        'track_buffer':29,
                        'track_thresh':0.7036
                    },
                    'MOT17-14-FRCNN':{
                        'track_buffer':28,
                        'track_thresh':0.5436
                    },
                }
        
        def set_args(args, video):
            if video not in setting.keys():
                return args
            for k,v in setting[video].items():
                args.__setattr__(k,v)
            return args
        
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        cfg = FullConfig()
        model_hmr, model_cfg = load_hmr2(self.args.checkpoint)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_hmr = model_hmr.to(device)
        model_hmr.eval()

        pose_predictor = Pose_transformer_v2(cfg, model_hmr)
        pose_predictor.load_weights(cfg.pose_predictor.weights_path)
        pose_predictor.to('cuda')
            
        tracker = MIXTracker_pose(cfg, self.args, pose_predictor)
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
        
            with torch.no_grad():
                # init tracker
                try:
                    frame_id = info_imgs[2].item()
                    video_id = info_imgs[3].item()
                    img_file_name = info_imgs[4]
                    video_name = img_file_name[0].split('/')[0]
                    
                    if video_name not in video_names:
                        video_names[video_id] = video_name
                    if frame_id == 1:
                        self.args = set_args(self.args, video_name)
                        if 'MOT17' in video_name:
                            self.args.alpha = 0.8778
                            self.args.iou_thresh = 0.2217
                            self.args.match_thresh = 0.7986
                        tracker.re_init(self.args)
                        if len(results) != 0:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                            write_results(result_filename, results)
                            results = []
                except:
                    frame_id = info_imgs[2][0].item()
                    video_id = info_imgs[3][0].item()
                    img_file_name = info_imgs[4][0]
                    video_name = img_file_name[0].split('/')[0]
                    if frame_id == 1:
                        tracker = MIXTracker_pose(cfg, self.args, pose_predictor)
                    if video_name not in video_names:
                        video_names[video_id] = video_name
                    if frame_id == 1:
                        self.args = set_args(self.args, video_name)
                        if 'MOT17' in video_name:
                            self.args.alpha = 0.8778
                            self.args.iou_thresh = 0.2217
                            self.args.match_thresh = 0.7986
                        tracker.re_init(self.args)
                        
                        if len(results) != 0:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                            write_results(result_filename, results)
                            results = []
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                with torch.no_grad():
                    outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/test/',info_imgs[4][0]))

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)
            _,_, img_height, img_width = origin_imgs.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            asd = get3d_human_features(cfg, model_hmr, outputs[0], frame, measurments, frame_id)
            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(asd, outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())#标记
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_oc_eiou(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):

        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker, MIXTracker_eiou

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter
        save_folder = 'output_frames'
        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        tracker = MIXTracker_eiou(det_thresh = self.args.track_thresh,args=self.args, iou_threshold=self.args.iou_thresh,
            asso_func='eiou', delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, max_age=self.args.track_buffer)
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if frame_id == 1:
                    tracker.re_init()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                        write_results_no_score(result_filename, results)
                        results = []
                # init tracker
                video_id = info_imgs[3].item()
                img_name = img_file_name[0].split("/")[2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))
            #frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/val/',info_imgs[4][0]))
            # Assuming results are updated for the current frame
            #draw_and_save_results(results[-1], save_folder, frame_id, frame)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_oc_reid(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):

        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker_reid

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        tracker = MIXTracker(det_thresh = self.args.track_thresh,args=self.args, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, max_age=self.args.track_buffer)
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if frame_id == 1:
                    tracker.re_init()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                        write_results_no_score(result_filename, results)
                        results = []
                # init tracker
                video_id = info_imgs[3].item()
                img_name = img_file_name[0].split("/")[2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_mixsort_oc(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):

        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        tracker = MIXTracker(det_thresh = self.args.track_thresh,args=self.args, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, max_age=self.args.track_buffer)
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if frame_id == 1:
                    tracker.re_init()
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                        write_results_no_score(result_filename, results)
                        results = []
                # init tracker
                video_id = info_imgs[3].item()
                img_name = img_file_name[0].split("/")[2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        from yolox.ocsort_tracker.ocsort import OCSort
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1


            
        tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia)
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                img_name = img_file_name[0].split("/")[2]
                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized 
                    stack of parameters on all datasets.
                """
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                        asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia)
                    if len(results) != 0:
                        try:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        except:
                            import pdb; pdb.set_trace()
                        write_results_no_score(result_filename, results)
                        results = []
                

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        
        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_ocsort_eiou(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        from yolox.ocsort_tracker.ocsort import OCSort, OCSort_eiou
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1


            
        tracker = OCSort_eiou(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia)
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                img_name = img_file_name[0].split("/")[2]
                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized 
                    stack of parameters on all datasets.
                """
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OCSort_eiou(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                        asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia)
                    if len(results) != 0:
                        try:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        except:
                            import pdb; pdb.set_trace()
                        write_results_no_score(result_filename, results)
                        results = []
                

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda())
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))
            frame = cv2.imread(os.path.join('/data2/zxd/sportsmot_publish/dataset/val/',info_imgs[4][0]))
            # Assuming results are updated for the current frame
            draw_and_save_results(results[-1], './output_frames', frame_id, frame)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                #vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area :
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
