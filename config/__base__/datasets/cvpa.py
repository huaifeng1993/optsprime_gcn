dataset_type = 'CVPADataset'
data_root = 'data/CVPA/'
batch_size=1
num_workers=1

data=dict(
    batch_size=batch_size,
    num_workers=num_workers,
    inputs_extkeys=[],
    train=dict(
        type=dataset_type,
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root
    )
)