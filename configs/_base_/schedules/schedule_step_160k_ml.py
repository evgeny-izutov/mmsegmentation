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
    by_epoch=False,
    iters=2000,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customstep',
    by_epoch=False,
    gamma=0.1,
    step=[100000, 130000],
    fixed='constant',
    fixed_iters=2000,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=4000,
    warmup_ratio=1e-2,
)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=160000
)
checkpoint_config = dict(
    by_epoch=False,
    interval=1000
)
evaluation = dict(
    interval=1000,
    metric='mIoU'
)

find_unused_parameters = True
