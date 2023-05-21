_base_=[
    "./__base__/datasets/cvpa.py",
    "./__base__/default_runtime.py"
]

optimizer = dict(type='Adam',
                 lr=5e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.0001)

lr_schedule = dict(
    type="StepLR",
    step_size=1000,
    gamma=0.1)

train_cfg = dict(max_epochs=1000,
                 print_freq=5,
                 val_step=5,
                 clip_norm=37.5
                 )

model=dict(
    type="GrapNodeCls",
    encoder=dict(
        type="SAGE",
        num_features=48,
        hidden_channels=100,
        num_classes=1,
        )
)