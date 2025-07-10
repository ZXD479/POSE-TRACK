import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import torch
import argparse
import sys
sys.path.append('.')
import numpy as np
from loguru import logger
import cv2


# Global
import time
from pose_pre.pose_trsanformerv2 import get3d_human_features
from yolox.mixsort_tracker.mixsort_tracker import Pose_Track
from pose_pre.base import FullConfig 
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from tqdm import tqdm
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full
def tlwh_to_x1y1x2y2(tlwh):  

    if tlwh.ndim != 2 or tlwh.shape[1] != 4:  
        raise ValueError("Input must be of shape (N, 4)")  
  
    x1 = tlwh[:, 0]  
    y1 = tlwh[:, 1]  
    w = tlwh[:, 2]  
    h = tlwh[:, 3]  
  
    x2 = x1 + w  
    y2 = y1 + h  
  
    return np.column_stack((x1, y1, x2, y2)) 
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("POSE-TRACK For Evaluation!")
    parser.add_argument("--output_dir", default="shiyan", type=str, help="Directory of saved results")
    parser.add_argument("--dataset_dir", default="/data2/zxd/sportsmot_publish/dataset/test/", type=str, help="Directory of dataset")
    parser.add_argument("--root_path", default="/home/zxd/project/Deep-EIoU-main/Deep-EIoU/", type=str)
    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.3, help='threshold for rejecting low appearance similarity reid matches')

    return parser

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def image_track(detections, pose_embedding, embeddings, sct_output_path, seq, pbar, args):
    cfg = FullConfig()
    model_hmr, _ = load_hmr2(DEFAULT_CHECKPOINT)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_hmr = model_hmr.to(device)
    model_hmr.eval()
    #model_hmr = None

    
    # Tracker
    tracker =  Pose_Track(cfg, args, frame_rate=30)
    
    results = []
    scale = min(1440/1280, 800/720)
    files = os.listdir(args.dataset_dir + seq + '/img1') 
    files = sorted(files)

    #renderer = Renderer(model_cfg, faces=model_hmr.smpl.faces)
    renderer = None
    for frame_id,det in enumerate(detections,1):

        frame = cv2.imread(args.dataset_dir + seq + '/img1/' + files[frame_id-1])
        det /= scale
        features = get3d_human_features(cfg, model_hmr, det,  [720, 1280, 1280, 0, 280], frame_id, pose_embedding[frame_id-1], renderer, frame)
        embs = embeddings[frame_id-1]

        if det is not None:

            embs = [e[0] for e in embs]
            embs = np.array(embs)

            trackerTimer.tic()
            online_targets = tracker.update(features, det, embs)
            trackerTimer.toc()
            results_tmp = []
            online_tlwhs = []
            online_ids = []
            online_scores = [] 
            for t in online_targets:
                tlwh = t.last_tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                vertical = False
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            results_tmp.append((frame_id, online_tlwhs, online_ids))

            timer.toc()
            # bboxes = tlwh_to_x1y1x2y2(np.array(online_tlwhs))
            # for bbox in bboxes:
            #     # 提取检测框的坐标
            #     x1, y1, x2, y2 = map(int, bbox[:4])  # 转换为整数

            #     # 在 frame 上绘制矩形框
            #     color = (0, 255, 0)  # 绿色框
            #     thickness = 2       # 框的线宽
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # # 保存带有检测框的图片
            # output_image_path = "output_with_boxes.jpg"
            # cv2.imwrite(output_image_path, frame)
        else:
            timer.toc()
        pbar.update(1)

    with open(sct_output_path, 'w') as f:
        f.writelines(results)
    logger.info(f"save SCT results to {sct_output_path}")
    
def main():
    
    args = make_parser().parse_args()
    data_path = args.root_path
    seq_path =  os.path.join(data_path,'detection/')
    data_folder = args.output_dir
    os.makedirs(os.path.join(data_folder), exist_ok=True)

    seqs = os.listdir(seq_path)
    seqs = [path.replace('.npy','') for path in seqs if path.endswith('.npy')]
    seqs.sort()
    with tqdm(total=94835, desc="Processing files") as pbar:
        for seq in seqs:   

            if not os.path.exists(os.path.join(data_path,'detection/','{}.npy'.format(seq))):
                continue

            pose_embedding = np.load(os.path.join(data_path,'pose_embedding/','{}.npy'.format(seq)),allow_pickle=True)
            
            detections = np.load(os.path.join(data_path,'detection/','{}.npy'.format(seq)),allow_pickle=True)
            
            embeddings = np.load(os.path.join(data_path,'embedding/','{}.npy'.format(seq)),allow_pickle=True)
                
            sct_output_path = os.path.join(data_folder,'{}.txt'.format(seq))
            
            # SCT tracking
            image_track(detections, pose_embedding, embeddings, sct_output_path, seq, pbar, args)
                
if __name__ == "__main__":
    main()
