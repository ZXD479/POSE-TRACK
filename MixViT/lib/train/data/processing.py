import torch
import torchvision.transforms as transforms
from MixViT.lib.utils import TensorDict
import MixViT.lib.train.data.processing_utils as prutils
import torch.nn.functional as F
from MixViT.lib.models.losses.target_classification import gaussian_label_function
from MixViT.lib.utils.box_ops import box_xywh_to_xyxy
import numpy as np


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class MixformerProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, output_feat_sz,
                 mode='pair', settings=None, train_score=False, generate_labels=False, sigma_factor=0.05, kernel_sz=3,
                 end_pad_if_even=False, radius=2, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.train_score = train_score
        # self.label_function_params = label_function_params
        self.out_feat_sz = output_feat_sz

        self.generate_labels = generate_labels
        self.sigma_factor = sigma_factor
        self.kernel_sz = kernel_sz
        self.end_pad_if_even = end_pad_if_even
        self.radius = radius

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        # num_proposals = self.proposal_params['boxes_per_frame']
        # proposal_method = self.proposal_params.get('proposal_method', 'default')

        # if proposal_method == 'default':
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou,
                                                             sigma_factor=sigma)
        # elif proposal_method == 'gmm':
        #     proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
        #                                                                      num_samples=num_proposals)
        #     gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1
        return proposals

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            data[s + '_images'+'_origin'] = torch.tensor(np.array(crops))
            data[s + '_anno'+'_origin'] = boxes[0].clone().detach()
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        # Generate labels
        if self.generate_labels:
            gt_boxes = data['search_anno'][0]  # shape (4) xywh, normalized
            # data['reg_labels'] = self._generate_regression_label(box_xywh_to_xyxy(gt_boxes),
            #                                                      self.out_feat_sz)  # (b, 4, w, h)
            # data['reg_mask'] = self._generate_regression_mask(gt_boxes, data['reg_labels'], self.radius,
            #                                                   self.out_feat_sz)  # (b, w, h)
            # data['cls_labels'] = self._generate_classification_label(gt_boxes, self.sigma_factor, self.kernel_sz,
            #                                                          self.out_feat_sz,
            #                                                          self.end_pad_if_even)  # (b, w, h)
            data['label'] = prutils.generate_gaussian(gt_boxes, self.out_feat_sz, self.output_sz['search'])

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


    def _generate_regression_label(self, boxes_xyxy, size=20):
        '''
        Generate a gt boxes map from a single gt bbox
        boxes_xyxy: gt box xyxy, shape (b, 4) normalized value in [0, 1)
        '''
        boxes_xyxy = boxes_xyxy * size
        x = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, 1, -1)
        y = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, -1, 1)

        l = (x - boxes_xyxy[:, 0].view(-1, 1, 1)).repeat(1, size, 1)  # (n, 1, size) --> (n, size, size)
        r = (boxes_xyxy[:, 2].view(-1, 1, 1) - x).repeat(1, size, 1)  # (n, 1, size) --> (n, size, size)
        t = (y - boxes_xyxy[:, 1].view(-1, 1, 1)).repeat(1, 1, size)  # (n, size, 1) --> (n, size, size)
        b = (boxes_xyxy[:, 3].view(-1, 1, 1) - y).repeat(1, 1, size)  # (n, size, 1) --> (n, size, size)

        reg_labels = torch.stack([l, r, t, b], dim=1)  # (n, 4, size, size)
        return reg_labels / size

    def _generate_regression_mask(self, box_xywh, reg_labels=None, radius=2, mask_size=20):
        """
        NHW format
        :param box_xywh: gt_box, shape (b, 4), norm
        :param radius:
        :param mask_size: size of feature map
        :return:
        """
        target_center = (box_xywh[:, 0:2] + 0.5 * box_xywh[:, 2:4]) * mask_size
        target_center = target_center.int().float()
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = k0 - target_center[:, 0].view(-1, 1, 1)
        d1 = k1 - target_center[:, 1].view(-1, 1, 1)
        dist = d0.abs() + d1.abs()

        rPos = radius + 0.1
        mask = torch.where(dist <= rPos, torch.ones_like(dist), torch.zeros_like(dist))  # (n, size, size)

        # remove the points outside the target box.
        if reg_labels is not None:
            mask = mask.unsqueeze(1).repeat(1, reg_labels.size(1), 1, 1)
            mask = torch.where(reg_labels <= 0, torch.zeros_like(mask), mask)
            mask = mask.min(dim=1)[0]  # (n, size, size)

        return mask

    def _generate_classification_label(self, box_xywh, sigma_factor, kernel_sz, feat_sz, end_pad_if_even=True):
        gauss_label = gaussian_label_function(box_xywh, sigma_factor, kernel_sz, feat_sz, end_pad_if_even)
        return gauss_label  # (b, feat_sz, feat_sz)
