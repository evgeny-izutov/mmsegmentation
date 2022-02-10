# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(
    grad_clip=dict(
        method='default',
        max_norm=40,
        norm_type=2
    )
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=True,
    iters=40,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customstep',
    by_epoch=True,
    gamma=0.1,
    step=[400, 500],
    fixed='constant',
    fixed_iters=40,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=1e-2,
)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=600
)
checkpoint_config = dict(
    by_epoch=True,
    interval=20
)
evaluation = dict(
    by_epoch=True,
    interval=20,
    metric='mIoU'
)

find_unused_parameters = True
