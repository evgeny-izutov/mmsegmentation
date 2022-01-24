_base_ = [
    '../_base_/models/fcn_litehr30_no-aggregator.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml_adam.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='FCNHead',
        in_channels=[40, 80, 160, 320],
        in_index=[0, 1, 2, 3],
        channels=sum([40, 80, 160, 320]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_aggregator=False,
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
