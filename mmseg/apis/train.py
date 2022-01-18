import random
import warnings
import os.path as osp

import numpy as np
import torch
import mmcv
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmcv.utils import build_from_cfg

from mmseg.core import DistEvalHook, EvalHook, CustomOptimizerHook, load_checkpoint, IterBasedEMAHook
from mmseg.datasets import build_dataloader, build_dataset, RepeatDataset
from mmseg.utils import get_root_logger
from mmseg.parallel import MMDataCPU
from mmseg.models import build_params_manager
from mmseg.models.losses import MarginCalibrationLoss


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def needed_collect_dataset_stat(cfg):
    for head_type in ['decode_head', 'auxiliary_head']:
        heads = cfg.model[head_type]
        if not isinstance(heads, (tuple, list)):
            heads = [heads]

        for head in heads:
            losses = head.loss_decode
            if not isinstance(losses, (tuple, list)):
                losses = [losses]
        
            for loss in losses:
                if loss.type == 'MarginCalibrationLoss':
                    return True

    return False


def set_dataset_stat(model, dataset_stat):
    for head_type in ['decode_head', 'auxiliary_head']:
        heads = getattr(model, head_type)
        if heads is None:
            continue

        if not isinstance(heads, nn.ModuleList):
            heads = [heads]
        
        for head in heads:
            for loss in head.loss_modules:
                if not isinstance(loss, MarginCalibrationLoss):
                    continue

                loss.set_margins(dataset_stat)


def collect_dataset_stat(dataset, tau=10.0, upsilon=1.0):
    # Source code: https://github.com/yutao1008/margin_calibration

    if isinstance(dataset, RepeatDataset):
        dataset = dataset.dataset
    
    num_classes = len(dataset.CLASSES)
    z = np.zeros((num_classes,))

    for item_id in range(len(dataset)):
        ann_info = dataset.get_ann_info(item_id)
        seg_map_file = osp.join(dataset.ann_dir, ann_info['seg_map'])
        gt_seg_map = mmcv.imread(seg_map_file, flag='unchanged', backend='pillow')
        if dataset.reduce_zero_label:
            assert dataset.ignore_index == 255
            gt_seg_map[gt_seg_map == 0] = 255
            gt_seg_map = gt_seg_map - 1
            gt_seg_map[gt_seg_map == 254] = 255

        mask = gt_seg_map != dataset.ignore_index
        labels = gt_seg_map[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    
    n_pixels = np.sum(z)
    
    rho_i0s, rho_0is = [], []
    for pixels in z:
        cls_prob = pixels / n_pixels
        bg_pixels = n_pixels - pixels
        rho_0i = tau * bg_pixels**0.5 / pixels
        rho_i0 = rho_0i * cls_prob * pixels**0.5 / (upsilon*bg_pixels - cls_prob * bg_pixels**0.5)
        rho_i0s.append(rho_i0)
        rho_0is.append(rho_0i)
    
    return np.array([rho_i0s,rho_0is])


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=False)
        for ds in dataset
    ]

    if needed_collect_dataset_stat(cfg):
        dataset_stat = collect_dataset_stat(dataset[0])
        set_dataset_stat(model, dataset_stat)

    if torch.cuda.is_available():
        # put model on gpus
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters
            )
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]),
                device_ids=cfg.gpu_ids
            )
    else:
        model = MMDataCPU(model)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    # build runner
    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning
        )
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta
        )
    )

    # prepare optimizer config
    if 'type' not in cfg.optimizer_config:
        optimizer_config = CustomOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register EMA hook
    ema_cfg = cfg.get('ema_config', None)
    if ema_cfg:
        runner.register_hook(IterBasedEMAHook(**ema_cfg))

    # register training hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None)
    )

    # register parameters manager hook
    params_manager_cfg = cfg.get('params_config', None)
    if params_manager_cfg is not None:
        runner.register_hook(build_params_manager(params_manager_cfg))

    # an ugly workaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), f'custom_hooks expect list type, but got ' \
                                               f'{type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), f'Each item in custom_hooks expects dict type, but got ' \
                                               f'{type(hook_cfg)}'

            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # load weights
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        load_checkpoint(
            model, cfg.load_from,
            logger=logger,
            force_matching=True,
            show_converted=True,
            ignore_keys=cfg.get('ignore_keys', None)
        )

    # run training
    runner.run(data_loaders, cfg.workflow)
