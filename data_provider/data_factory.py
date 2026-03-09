from data_provider.data_loader import Dataset_zero_shot
from torch.utils.data import DataLoader, ConcatDataset

data_dict = {
    'zero_shot': Dataset_zero_shot,
}


def data_provider(args, flag, vali=False):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    train_data_path = args.train_data_path
    if flag == 'train' or flag == 'val':
        datasets = []
        for data_path in train_data_path:
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                data_path=data_path,
                target=args.target,
            )
            datasets.append(data_set)
        datasets = ConcatDataset(datasets)
        print(flag, len(datasets))

        data_loader = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return datasets, data_loader

    else: #flag == 'test'
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            data_path=args.test_data_path,
            target=args.target,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

