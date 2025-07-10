import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import sys
sys.path.append('.')
import numpy as np
from loguru import logger
import cv2
#from tracker.Deep_EIoU import Deep_EIoU
from yolox.mixsort_tracker.mixsort_tracker import Deep_EIoU_pose, Deep_EIoU_dance_pose
# Global
import time
from pose_pre.pose_trsanformerv2 import Pose_transformer_v2, smpl_to_pose_camera_vector, get3d_human_features
from pose_pre.base import FullConfig 
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from tqdm import tqdm
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full

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
    parser = argparse.ArgumentParser("DeepEIoU For Evaluation!")

    parser.add_argument("--root_path", default="/home/zxd/project/MixSort-main/", type=str)
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

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

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

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

def image_track(detections, pose_embedding, sct_output_path, seq, pbar, args):



    cfg = FullConfig()

    # model_hmr, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model_hmr = model_hmr.to(device)
    # model_hmr.eval()
    model_hmr = None
    # pose_predictor = Pose_transformer_v2(cfg, model_hmr)
    # pose_predictor.load_weights(cfg.pose_predictor.weights_path)
    # pose_predictor.to('cuda')    
    
    # Tracker
    tracker =  Deep_EIoU_dance_pose(cfg, args,  frame_rate=30)
    
    results = []
    
    num_frames = len(detections)
    save_folder = './output_frames'
    scale = min(1440/1920, 800/1080)
    files = os.listdir('/data2/zxd/DanceTrack/val/' + seq + '/img1') 
    files = sorted(files)
    # if seq == 'dancetrack0019':
    #     print('s')
    #renderer = Renderer(model_cfg, faces=model_hmr.smpl.faces)
    renderer = None
 
    for frame_id,det in enumerate(detections,1):
        if frame_id == 1:
            flag = 0
        #if seq == 'dancetrack0019':
        frame = cv2.imread('/data2/zxd/DanceTrack/val/' + seq + '/img1/' + files[frame_id-1])
        img_height, img_width, _ = frame.shape
        new_image_size            = max(img_height, img_width)
        top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
        measurments               = [img_height, img_width, new_image_size, left, top]  
        if img_height == 720:
            scale = min(1440/1280, 800/720)
        #else:
            #frame = None
        det /= scale
        asd, flag = get3d_human_features(cfg, model_hmr, det, [800, 1440, 1440, 0, 320], frame_id, flag, pose_embedding[frame_id-1], renderer)
        #embs = embeddings[frame_id-1]
        # if frame_id == 1:
        #     all_pose = []
        # poses = []
        # for obj in asd:
        #     # 假设每个物体都有一个 'pose' 属性
        #     poses.append(obj['pose'])
        # all_pose.append(poses)
        # if True:
        #     pbar.update(1)
        #     continue
        if det is not None:

            # embs = [e[0] for e in embs]
            # embs = np.array(embs)

            trackerTimer.tic()
            online_targets = tracker.update(asd, det, flag, frame_id)
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
            #draw_and_save_results(results_tmp[-1], save_folder, frame_id, frame)
            timer.toc()

        else:
            timer.toc()
        pbar.update(1)
        # if frame_id % 100 == 0:
        #     logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))
    # np.save('./pose_embedding/'+ seq+'.npy',all_pose)
    with open(sct_output_path, 'w') as f:
        f.writelines(results)
    logger.info(f"save SCT results to {sct_output_path}")
    
def main():
    
    args = make_parser().parse_args()
    data_path = args.root_path
    seq_path =  os.path.join(data_path,'pose_dance_bboxes/')
    data_folder = 'TrackEval/data/trackers/mot_challenge/dance-val/data'
    os.makedirs(os.path.join(data_folder), exist_ok=True)
    save_folder = './output_frames/'    
    seqs = os.listdir(seq_path)
    seqs = [path.replace('.npy','') for path in seqs if path.endswith('.npy')]
    seqs.sort()
    with tqdm(total=25508, desc="Processing files") as pbar:
        for seq in seqs:                
            #logger.info('Processing seq {}'.format(seq))

            if not os.path.exists(os.path.join(data_path,'pose_dance_bboxes/','{}.npy'.format(seq))):
                continue

            pose_embedding = np.load(os.path.join(data_path,'pose_dance_embedding/','{}.npy'.format(seq)),allow_pickle=True)
            
            detections = np.load(os.path.join(data_path,'pose_dance_bboxes/','{}.npy'.format(seq)),allow_pickle=True)
            
            #embeddings = np.load(os.path.join(data_path,'embedding/','{}.npy'.format(seq)),allow_pickle=True)
                
            sct_output_path = os.path.join(data_folder,'{}.txt'.format(seq))
            
            # SCT tracking
            image_track(detections, pose_embedding,  sct_output_path, seq, pbar, args)
                
if __name__ == "__main__":
    main()
