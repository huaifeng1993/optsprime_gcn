optimizer = dict(type='SGD',
                 lr=0.02,
                 momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_schedule = dict(
    type="CosineAnnealingLR",
    T_max=80,
    eta_min=0.0000001)