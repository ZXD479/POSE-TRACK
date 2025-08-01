import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from MixViT.lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2k, SportsMOT, MOT17, DanceTrack, MOT20, SoccerNet
from MixViT.lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from MixViT.lib.train.data import sampler, opencv_loader, processing, LTRLoader
import MixViT.lib.train.data.transforms as tfm
from MixViT.lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["SoccerNet_train","SoccerNet_test","DanceTrack_train","DanceTrack_val","SportsMOT_train","SportsMOT_val","SportsMOT_mix","SportsMOT_test","MOT17-train","MOT17-train_half","MOT20-train","MOT17-val_half","LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET", "TNL2k"]
        if "SoccerNet" in name:
            datasets.append(SoccerNet(settings.env.soccernet_dir,settings.env.soccernet_anno_dir, split=name.split('_')[1], image_loader=image_loader))
        if "Sports" in name:
            datasets.append(SportsMOT(settings.env.sportsmot_dir,settings.env.sportsmot_anno_dir, split=name.split('_')[1], image_loader=image_loader))
        if "MOT17" in name:
            datasets.append(MOT17(settings.env.mot17_dir,settings.env.mot17_anno_dir, split=name.split('-')[1], image_loader=image_loader))
        if "MOT20" in name:
            datasets.append(MOT20(settings.env.mot20_dir,settings.env.mot20_anno_dir, split=name.split('-')[1], image_loader=image_loader))
        if "DanceTrack" in name:
            datasets.append(DanceTrack(settings.env.dancetrack_dir,settings.env.dancetrack_anno_dir, split=name.split('_')[1], image_loader=image_loader))
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "TNL2k":
            datasets.append(TNL2k(settings.env.tnl2k_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    print("sampler_mode", sampler_mode)
    output_feat_sz = output_sz['search'] // 16

    data_processing_train = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           transform=transform_train,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           generate_labels=True,
                                                           train_score=train_score,
                                                           output_feat_sz=output_feat_sz,)
                                                           #generate_labels=settings.generate_sep_labels)

    data_processing_val = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_val,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score,
                                                         output_feat_sz=output_feat_sz,
                                                         generate_labels=True)
                                                         #generate_labels=settings.generate_sep_labels)


    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    freeze_stage0 = getattr(cfg.TRAIN, "FREEZE_STAGE0", False)
    # freeze_12layers = getattr(settings, "freeze_12layers", False)
    # freeze_stage0 = True
    freeze_partial_layers = False
    cosin_lr = False
    if train_score:
        print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "score" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "score" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_stage0:
        print("Freeze Stage0 of MixFormer backbone.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if (("stage2" in n or "stage1" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

        for n, p in net.named_parameters():
            if "stage2" not in n and "box_head" not in n and "stage1" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

    elif freeze_partial_layers:
        print("Freeze 6 layers of MixFormer-vit-large backbone.")
        for n, p in net.named_parameters():
            for i in range(6):
                if 'blocks.{}.'.format(i) in n:
                    p.requires_grad = False
            if 'patch_embed' in net.named_parameters():
                p.requires_grad = False
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if (("backbone" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    elif cosin_lr: # train network except for score prediction module
        for n, p in net.named_parameters():
            if "score" in n:
                p.requires_grad = False

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        ]
        import math
        min_lr = cfg.TRAIN.MIN_LR
        # lr = cfg.TRAIN.LR
        lr = cfg.TRAIN.LR * 0.1

        for block_id in range(12):
            # for n, p in net.named_parameters():
            #     if "backbone.blocks.{}".format(block_id) in n and p.requires_grad:
            #         print(block_id, n)
            param_dicts.append({
                "params": [p for n, p in net.named_parameters() if "backbone.blocks.{}.".format(block_id) in n and p.requires_grad],
                # "lr": min_lr + (lr - min_lr) * math.sin(0.5 * math.pi * block_id / 11)
                "lr": lr - (lr - min_lr) * math.cos(0.5 * math.pi * block_id / 11)
                # "lr": min_lr + (lr - min_lr) * math.cos(0.5 * math.pi * block_id / 11)
            },)
            # print("block-{}-lr: {}".format(block_id, min_lr + (lr - min_lr) * math.cos(0.5 * math.pi * block_id / 11)))
            print("block-{}-lr: {}".format(block_id, lr - (lr - min_lr) * math.cos(0.5 * math.pi * block_id / 11)))

    else: # train network except for score prediction module
        for n, p in net.named_parameters():
            if "score" in n:
                p.requires_grad = False

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
                "lr_scale": cfg.TRAIN.BACKBONE_MULTIPLIER
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.LR_DROP_EPOCH,
                                                            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler