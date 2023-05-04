dataset_type = 'PyGPPDataset'
data_root = 'data/original_datasets/'
batch_size=1
num_workers=1

data=dict(
    batch_size=batch_size,
    num_workers=num_workers,
    inputs_extkeys=[],
    train=dict(
        type=dataset_type,
        name="ogbg-molesol",
        root=data_root,
        split="train",
        transform=["initialize_edge_weight"],
    ),
    val=dict(
        type=dataset_type,
        name="ogbg-molesol",
        root=data_root,
        split="valid",
        transform=["initialize_edge_weight"]
    )
)