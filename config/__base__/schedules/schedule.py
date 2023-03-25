optimizer = dict(type='SGD',
                 lr=0.02,
                 momentum=0.9,
                 weight_decay=0.0001)

lr_schedule = dict(
    type="CosineAnnealingLR",
    T_max=80,
    eta_min=0.0000001)

train_cfg = dict(max_epochs=200,
                 print_freq=5,
                 val_step=10,
                 clip_norm=37.5
                 )
