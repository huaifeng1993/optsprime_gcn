_base_=[
    "./__base__/datasets/adg.py",
    "./__base__/schedules/schedule.py",
    "./__base__/default_runtime.py"
]

model=dict(
    type="GInfoMinMax",
    encoder=dict(
        type="MoleculeEncoder",
        emb_dim=300,
        num_gc_layers=5,
        drop_ratio="0.0",
        pooling_type="standard",
        is_infograph=False,
        ),
    decoder=dict(
        type="LinearHead",
        input_proj_dim=300,
        proj_hidden_dim=300,
    )
)