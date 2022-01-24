_base_ = [
    '../_base_/models/fcn_litehr18_no-aggregator.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml_adam.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    decode_head=[
        dict(type='FCNHead',
             in_channels=[40, 80, 160, 320],
             in_index=[0, 1, 2, 3],
             input_transform='multiple_select',
             channels=40,
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_aggregator=True,
             enable_out_norm=False,
             loss_decode=[
                 dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_jitter_prob=0.01,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
        dict(type='OCRHead',
             in_channels=[40, 80, 160, 320],
             in_index=[0, 1, 2, 3],
             input_transform='multiple_select',
             channels=40,
             ocr_channels=40,
             sep_conv=True,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_aggregator=True,
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
                      gamma=0.0,
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

find_unused_parameters = False
