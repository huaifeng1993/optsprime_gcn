optimizer = dict(type='Adam',
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.0001)

lr_schedule = dict(
    type="CosineAnnealingLR",
    T_max=200,
    eta_min=0.0000001)

train_cfg = dict(max_epochs=200,
                 print_freq=5,
                 val_step=5,
                 clip_norm=37.5
                 )
