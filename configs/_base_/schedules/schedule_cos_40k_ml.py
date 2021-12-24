# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(
    grad_clip=dict(
        # method='adaptive',
        # clip=0.2,
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
    policy='customcos',
    by_epoch=False,
    periods=[34000],
    min_lr_ratio=1e-2,
    alpha=1.2,
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
    max_iters=40000
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
