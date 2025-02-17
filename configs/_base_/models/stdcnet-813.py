# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='STDCNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            backbone='STDCNet813'
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=-1,
        channels=256,
        input_transform=None,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
