_base_ = [
    '../_base_/models/fcn_litehrxv3_no-aggregator.py', '../_base_/datasets/kvasir_extra.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml_adam.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    decode_head=[
        dict(type='FCNHead',
             in_channels=[18, 60, 80, 160, 320],
             in_index=[0, 1, 2, 3, 4],
             input_transform='multiple_select',
             channels=60,
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_aggregator=True,
             aggregator_min_channels=60,
             aggregator_merge_norm=None,
             aggregator_use_concat=False,
             enable_out_norm=False,
             enable_loss_equalizer=True,
             loss_decode=[
                 dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_jitter_prob=0.01,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=4.0),
                 dict(type='GeneralizedDiceLoss',
                      smooth=1.0,
                      gamma=5.0,
                      alpha=0.5,
                      beta=0.5,
                      focal_gamma=1.0,
                      loss_jitter_prob=0.01,
                      loss_weight=4.0),
             ]),
        dict(type='OCRHead',
             in_channels=[18, 60, 80, 160, 320],
             in_index=[0, 1, 2, 3, 4],
             input_transform='multiple_select',
             channels=60,
             ocr_channels=60,
             sep_conv=True,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_aggregator=True,
             aggregator_min_channels=60,
             aggregator_merge_norm=None,
             aggregator_use_concat=False,
             enable_out_norm=True,
             loss_decode=[
                 dict(type='AMSoftmaxLoss',
                      scale_cfg=dict(
                          type='PolyScalarScheduler',
                          start_scale=30,
                          end_scale=5,
                          num_iters=30000,
                          power=1.2
                      ),
                      margin_type='cos',
                      margin=0.5,
                      gamma=2.0,
                      t=1.0,
                      target_loss='ce',
                      pr_product=False,
                      conf_penalty_weight=dict(
                          type='PolyScalarScheduler',
                          start_scale=0.2,
                          end_scale=0.15,
                          num_iters=20000,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
    ],
    train_cfg=dict(
        mix_loss=dict(
            enable=False,
            weight=0.1
        ),
        loss_reweighting=dict(
            weights={'decode_0.loss_seg': 0.9,
                     'decode_1.loss_seg': 1.0},
            momentum=0.1
        ),
    ),
)
evaluation = dict(
    metric='mDice',
)
