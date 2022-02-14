_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_40k.py'
]

model = dict(
    # pretrained='open-mmlab://msra/hrnetv2_w18_small',
    pretrained=None,
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2))
        )
    ),
    decode_head=dict(
        num_classes=2,
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
    test_cfg=dict(
        mode='slide',
        crop_size=(1024, 1024),
        stride=(680, 680)
    ),
)
evaluation = dict(
    metric='mDice',
)
