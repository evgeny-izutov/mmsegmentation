# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-3,
    eps=1e-08,
    weight_decay=0.0
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
    iters=0,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customstep',
    by_epoch=False,
    gamma=0.1,
    step=[20000, 30000],
    warmup='cos',
    warmup_iters=6000,
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
