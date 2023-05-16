_base_=[
    "./__base__/datasets/cvpa.py",
    "./__base__/default_runtime.py"
]
optimizer = dict(type='Adam',
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.0001)

lr_schedule = dict(
    type="CosineAnnealingLR",
    T_max=200,
    eta_min=0.0000001)

train_cfg = dict(max_epochs=1000,
                 print_freq=5,
                 val_step=5,
                 clip_norm=37.5
                 )
model=dict(
    type="GrapNodeCls",
    encoder=dict(
        type="MLP",
        num_features=48,
        hidden_channels=100,
        proj_channels=100,
        ),
    decoder=dict(
        type="LinearHead",
        input_proj_dim=100,
        proj_hidden_dim=1,
    )
)