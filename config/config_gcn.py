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
model=dict(
    type="GrapNodeCls",
    encoder=dict(
        type="GCN",
        num_features=10,
        hidden_channels=64,
        num_classes=2
        )
)