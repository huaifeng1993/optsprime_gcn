dataset_type = 'GermanDataset'
data_root = 'data/Genman/'
batch_size=1
num_workers=1

data=dict(
    batch_size=batch_size,
    num_workers=num_workers,
    train=dict(
        type=dataset_type,
        data_root="data/Genman"
    ),
    val=dict(
        type=dataset_type,
        data_root="data/Genman"
    )
)