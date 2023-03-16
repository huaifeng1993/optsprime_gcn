dataset_type = 'GermanDataset'
data_root = 'data/Genman/'
data=dict(
    dataset_type=dataset_type,
    train=dict(
        type=dataset_type,
        data_root="data/Genman"
    ),
    val=dict(
        type=dataset_type,
        data_root="data/Genman"
    )
)