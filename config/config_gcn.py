_base_=[
    "./__base__/datasets/german.py",
    "./__base__/schedules/schedule.py",
    "./__base__/default_runtime.py"
]

model=dict(
    type="GrapNodeCls",
    encoder=dict(
        type="GCN",
        num_features=28,
        hidden_channels=64,
        num_classes=2
        )
)