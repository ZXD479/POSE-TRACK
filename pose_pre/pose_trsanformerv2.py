import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from pose_pre.smpl_head import SMPLHead
from omegaconf import OmegaConf
from torch import nn
import copy
from yolox.utils.boxes import expand_bbox_to_aspect_ratio, get_croped_image, get_croped_image1
import os
import cv2
from hmr2.utils.renderer import cam_crop_to_full
from thop import profile

def get_pose_distance(track_pose, detect_pose):
    """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`.""" 
    track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        # remove additional dimension used for encoding location (last 3 elements)
    track_pose = track_pose[:, :-3]
    detect_pose = detect_pose[:, :-3]

    if len(track_pose) == 0 or len(detect_pose) == 0:
        return np.zeros((len(track_pose), len(detect_pose)))
    track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
    r2 = -2. * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2
def smpl_to_pose_camera_vector(smpl_params):
    # convert smpl parameters to camera to pose_camera_vector for smoothness.
    global_orient_  = smpl_params['global_orient'].reshape(1, -1) # 1x3x3 -> 9
    body_pose_      = smpl_params['body_pose'].reshape(1, -1) # 23x3x3 -> 207
    shape_          = smpl_params['betas'].reshape(1, -1) # 10 -> 10
    # loca_           = copy.deepcopy(camera.view(1, -1)) # 3 -> 3
    # loca_[:, 2]     = loca_[:, 2]/200.0
    #pose_embedding  = np.concatenate((global_orient_, body_pose_, shape_, loca_.cpu().numpy()), 1)
    pose_embedding  = np.concatenate((global_orient_, body_pose_, shape_), 1)
    return pose_embedding
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe          = torch.zeros(length, d_model)
    position    = torch.arange(0, length).unsqueeze(1)
    div_term    = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim    = dim_head *  heads
        project_out  = not (heads == 1 and dim_head == dim)

        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask_all):
        qkv          = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v      = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots         = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        masks_np     = mask_all[0]
        masks_bert   = mask_all[1]
        BS           = masks_np.shape[0]
        masks_np     = masks_np.view(BS, -1)
        masks_bert   = masks_bert.view(BS, -1)
        
        masks_np_    = rearrange(masks_np, 'b i -> b () i ()') * rearrange(masks_np, 'b j -> b () () j')
        masks_np_    = masks_np_.repeat(1, self.heads, 1, 1)
        
        masks_bert_  = rearrange(masks_bert, 'b i -> b () () i')
        masks_bert_  = masks_bert_.repeat(1, self.heads, masks_bert_.shape[-1], 1)
                
        dots[masks_np_==0]   = -1e3
        dots[masks_bert_==1] = -1e3
        
        del masks_np, masks_np_, masks_bert, masks_bert_
        
        attn    = self.attend(dots)
        attn    = self.dropout(attn)

        out     = torch.matmul(attn, v)
        out     = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask_np):
        for attn, ff in self.layers:
            x_          = attn(x, mask_all=mask_np) 
            x           = x + self.drop_path(x_)
            x           = x + self.drop_path(ff(x))
            
        return x


    def forward(self, data, mask_type="random"):
        
        # prepare the input data and masking
        data, has_detection, mask_detection = self.bert_mask(data, mask_type)

        # encode the input pose tokens
        pose_   = data['pose_shape'].float()
        pose_en = self.pose_shape_encoder(pose_)
        x       = pose_en
        
        # mask the input tokens
        x[mask_detection[:, :, :, 0]==1] = self.mask_token

        BS, T, P, dim = x.size()
        x = x.view(BS, T*P, dim)

        # adding 2D posistion embedding
        # x = x + self.pos_embedding[None, :, :self.cfg.frame_length, :self.cfg.max_people].reshape(1, dim, self.cfg.frame_length*self.cfg.max_people).permute(0, 2, 1)
        
        x    = x + self.pos_embedding_learned1
        x    = self.transformer1(x, [has_detection, mask_detection])

        x = x.transpose(1, 2)
        x = self.conv_en(x)
        x = self.conv_de(x)
        x = x.transpose(1, 2)
        x = x.contiguous()

        x                = x + self.pos_embedding_learned2
        has_detection    = has_detection*0 + 1
        mask_detection   = mask_detection*0
        x    = self.transformer2(x, [has_detection, mask_detection])
        x = torch.concat([self.class_token.repeat(BS, self.cfg.max_people, 1), x], dim=1)
        

        return x, 0


class Pose_transformer_v2(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(Pose_transformer_v2, self).__init__()
        
        self.phalp_cfg = cfg

        # load a config file
        self.cfg = OmegaConf.load(self.phalp_cfg.pose_predictor.config_path).configs
        self.cfg.max_people = 1
        
        self.mean_, self.std_ = np.load(self.phalp_cfg.pose_predictor.mean_std, allow_pickle=True)
        self.mean_            = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_             = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        self.mean_, self.std_ = torch.tensor(self.mean_), torch.tensor(self.std_)
        self.mean_, self.std_ = self.mean_.float(), self.std_.float()
        self.mean_, self.std_ = self.mean_.unsqueeze(0), self.std_.unsqueeze(0)   
        self.register_buffer('mean', self.mean_)
        self.register_buffer('std', self.std_)
        
        self.smpl = phalp_tracker.smpl
            

def detction(detection_data):
        detection_data           = detection_data
        tlwh                     = np.asarray(detection_data['bbox'], dtype=np.float64)

        image_size                    = detection_data['size']
        img_height, img_width         = float(image_size[0]), float(image_size[1])
        new_image_size                = max(img_height, img_width)
        delta_w                       = new_image_size - img_width
        delta_h                       = new_image_size - img_height
        top, _, left, _               = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
        xy                       = [((tlwh[0]+tlwh[2])/2.0+left)/new_image_size, ((tlwh[1]+tlwh[3])/2.0+top)/new_image_size]
        detection_data['xy']     = xy
        detection_data['scale']  = float(max(detection_data['scale']))/new_image_size
        return detection_data
def get3d_human_features(cfg, model_hmr, detection, measurments, t_,  pose_embedding_ = None, render= None, frame=None):

    detection = torch.tensor(detection).cpu()
    #detection = detection.cpu()
    bbox = detection[:, :4]
    bbox_pad = expand_bbox_to_aspect_ratio(bbox, (192, 256))
    #score = detection[:, 4] * detection[:, 5]
    score = torch.ones((bbox.size(0),))
    NPEOPLE = len(score)

    if(NPEOPLE==0): return []
    img_height, img_width, new_image_size, left, top = measurments                
    ratio = 1.0/int(new_image_size)*cfg.render.res
    image_list = []
    center_list = []
    scale_list = []

    selected_ids = []
    for p_ in range(NPEOPLE):
            #continue

        image_tmp, center_pad, scale_pad = get_croped_image(frame, bbox[p_], bbox_pad[p_])#原来是bbox_pad
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

        # pred_smpl_params = hmar_out['pred_smpl_params']
        # pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]




        flops, params = profile(model_hmr, inputs=(image_list.cuda(),))
        print(f"FLOPs: {flops / 1e9} GFLOPs")
        print(f"Parameters: {params / 1e6} M")

        pose_embedding = []
        for i in range(BS): 
            #pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i])
            pose_embedding.append(torch.from_numpy(pose_embedding_[i]))
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