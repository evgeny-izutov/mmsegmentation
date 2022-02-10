_base_ = [
    '../_base_/models/fcn_litehr18_s2.py', '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='FCNHead',
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=1,
        num_convs=0,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_out_norm=True,
        loss_decode=[
            dict(type='AMSoftmaxLoss',
                 scale_cfg=dict(
                     type='ConstantScalarScheduler',
                     scale=10.0
                 ),
                 margin_type='cos',
                 margin=0.5,
                 gamma=0.0,
                 t=1.0,
                 target_loss='ce',
                 pr_product=False,
                 conf_penalty_weight=0.085,
                 loss_jitter_prob=0.01,
                 sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                 loss_weight=1.0),
        ]
    ),
    train_cfg=dict(
        mix_loss=dict(enable=False, weight=0.1)
    ),
    # test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(680, 680))
)
evaluation = dict(
    metric='mDice'
)

find_unused_parameters = True
