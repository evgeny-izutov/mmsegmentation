_base_ = [
    '../_base_/models/fcn_litehr18_no-aggregator.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml_adam.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='FCNHead',
        in_channels=[40, 80, 160, 320],
        in_index=[0, 1, 2, 3],
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
        enable_out_norm=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_jitter_prob=0.01,
                sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                loss_weight=10.0
            ),
        ]
    ),
    train_cfg=dict(
        mix_loss=dict(
            enable=False,
            weight=0.1
        ),
    ),
)
evaluation = dict(
    metric='mDice',
)
