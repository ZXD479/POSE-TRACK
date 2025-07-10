import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append('.')
import argparse
import numpy as np
import os.path as osp
import time
import cv2

import torch
from moviepy.editor import VideoFileClip
from loguru import logger
from reid.torchreid.utils.feature_extractor import FeatureExtractor
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.byte_tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from pose_pre.pose_trsanformerv2 import smpl_to_pose_camera_vector, detction
from pose_pre.base import FullConfig 


from typing import Optional, Tuple

import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from yolox.utils.boxes import expand_bbox_to_aspect_ratio, get_croped_image
from yolox.mixsort_tracker.mixsort_tracker import Pose_Track

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images1', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    parser.add_argument("--demo", default="video", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--alpha",type=float,default=0.6,help='fuse parameter')
    parser.add_argument("--radius",type=int,default=0,help='radius for computing similarity')
    parser.add_argument("--iou_thresh",type=float,default=0.3)
    parser.add_argument("--script",type=str,default='mixformer_deit')
    parser.add_argument("--config",type=str,default='track')
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--path", default="videos/basketball.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result",default=True,action="store_true",help="whether to save the inference result of image/video",)
    # exp file
    parser.add_argument("-f","--exp_file",default="exps/example/mot/yolox_x_sportsmot.py",type=str,help="pls input your expriment description file",)
    parser.add_argument("-c", "--ckpt", default='pretrained/yolox_x_sports_train.pth.tar', type=str, help="ckpt for eval")
    parser.add_argument("--device",default="gpu",type=str,help="device to run our model, can either be cpu or gpu",)
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16",dest="fp16",default=False,action="store_true",help="Adopting mix precision evaluating.",)
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true",help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt",dest="trt",default=False,action="store_true",help="Using TensorRT model for testing.",)
    # tracking args
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

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cfg,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.cfg = cfg

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    def get3d_human_features(cfg, model_hmr, detection, image, measurments, t_,  pose_embedding_ = None, render= None):

        detection = torch.tensor(detection).cpu()
        #detection = detection.cpu()
        bbox = detection[:, :4]
        bbox_pad = expand_bbox_to_aspect_ratio(bbox, (192, 256))
        score = detection[:, 4] * detection[:, 5]
        #score = torch.ones((bbox.size(0),))
        NPEOPLE = len(score)

        if(NPEOPLE==0): return []
        img_height, img_width, new_image_size, left, top = measurments                

        image_list = []
        center_list = []
        scale_list = []

        selected_ids = []
        for p_ in range(NPEOPLE):
                #continue

            image_tmp, center_pad, scale_pad = get_croped_image(image, bbox[p_], bbox_pad[p_])#原来是bbox_pad
            image_list.append(image_tmp)#裁剪图片，掩码
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            selected_ids.append(p_)
        if(len(image_list)==0): return []
        image_list = torch.stack(image_list, dim=0)
        BS = image_list.size(0)



        #     center_pad, scale_pad = get_croped_image1(bbox[p_], bbox_pad[p_])#原来是bbox_pad

        #     center_list.append(center_pad)
        #     scale_list.append(scale_pad)
        #     selected_ids.append(p_)
        # BS = len(center_list)



        with torch.no_grad():
            extra_args      = {}
            hmar_out        = model_hmr(image_list.cuda(), **extra_args) #得到参数

            pred_smpl_params = hmar_out['pred_smpl_params']
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]

            pose_embedding = []
            for i in range(BS): 
                pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i])
                pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
            pose_embedding = torch.stack(pose_embedding, dim=0)


        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                                "bbox"            : np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                                "conf"            : score[p_], 
                                "pose"            : pose_embedding[i].numpy(), 
                                "center"          : center_list[i],
                                "scale"           : scale_list[i],
                                "size"            : [img_height, img_width],
                                "img_name"        : "basketball_" + str(t_),
                                "time"            : t_,
                                "ground_truth"    : 1,

                            }
            detection_data_list.append(detction(detection_data))

        return detection_data_list



def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:## x1y1x2y2:outputs[0][:, :4]    scores = output_results[:, 4] * output_results[:, 5]
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        #np.save('bbox.npy', online_tlwhs)
        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            #cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(cfg, extractor, predictor, vis_folder, current_time, args):
    os.makedirs(args.out_folder, exist_ok=True)
    model_hmr, model_cfg = load_hmr2(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_hmr = model_hmr.to(device)
    model_hmr.eval()

    renderer = Renderer(model_cfg, faces=model_hmr.smpl.faces)
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    vid_writer_3d = cv2.VideoWriter(
        "3d_basketball.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = Pose_Track(cfg, args, None, frame_rate=30)
    #tracker = MIXTracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            img_height, img_width, _  = frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            det = outputs[0].cpu().detach().numpy()
            scale = min(1440/width, 800/height)
            det /= scale
            asd = predictor.get3d_human_features(model_hmr, det, frame, measurments, frame_id)
            
            if det is not None:
                # for bbox in det:
                #     # 提取检测框的坐标
                #     x1, y1, x2, y2 = map(int, bbox[:4])  # 转换为整数

                #     # 在 frame 上绘制矩形框
                #     color = (0, 255, 0)  # 绿色框
                #     thickness = 2       # 框的线宽
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # # 保存带有检测框的图片
                # output_image_path = "output_with_boxes1.jpg"
                # cv2.imwrite(output_image_path, frame)
 
                cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_,_,_ in det]
                
                embs = extractor(cropped_imgs)
                embs = embs.cpu().detach().numpy()

                online_targets = tracker.update(asd, det, embs)
                online_tlwhs = []
                online_ids = []
                track_id = []
                online_center = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        track_id.append(t.track_id)
                        online_center.append([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time)
                #cv2.imwrite('asd.png', online_im)
            else:
                timer.toc()
                online_im = img_info['raw_img']

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
            dataset = ViTDetDataset(model_cfg, frame, tlwh_to_x1y1x2y2(np.array(online_tlwhs)))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=False, num_workers=0)
            all_verts = []
            all_cam_t = []
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model_hmr(batch)
                pred_cam = out['pred_cam']
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    #img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()
                    ########################
                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )

                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                    if args.side_view:
                        side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                        final_img = np.concatenate([final_img, side_img], axis=1)

                    if args.top_view:
                        top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                top_view=True)
                        final_img = np.concatenate([final_img, top_img], axis=1)

                    #cv2.imwrite(os.path.join(args.out_folder, f'img_{frame_id}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, track_id, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

                # Overlay image
                input_img = frame.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(args.out_folder, f'img_{frame_id}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
                
                vid_writer_3d.write(cv2.convertScaleAbs(255*input_img_overlay[:, :, ::-1]))
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        input_file = "videos/basketball.mp4"  
        output_file = "3d_basketball.mp4"  
        
        # 读取视频文件  
        video = VideoFileClip(input_file)  
        
        # 将视频转换为指定格式并保存，同时设置编码器为libx264  
        video.write_videofile(output_file, codec="libx264")
@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


#@hydra.main(version_base="1.2", config_name="config")
def main(exp, args):

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    trt_file = None
    decoder = None
    cfg = FullConfig()
    predictor = Predictor(model, exp, cfg, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    #log.info("Loading Predictor model...")
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = 'checkpoints/sports_model.pth.tar-60',
        device='cuda')       
    if args.demo == "image":
        image_demo(cfg, predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(cfg, extractor, predictor, vis_folder, current_time, args)




if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
