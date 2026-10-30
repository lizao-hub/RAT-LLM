from data_provider.data_loader import Dataset_Custom, Dataset_Zero_shot
from torch.utils.data import DataLoader, ConcatDataset

data_dict = {
    'custom': Dataset_Custom,       # 一个 CSV 按时间 7:1:2 切 train/val/test
    'zero_shot': Dataset_Zero_shot,  # 一个 CSV 全部行都当测试集，只接受 flag='test'
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = True if (flag == 'train' or flag == 'TRAIN') else False
    batch_size = args.batch_size
    # freq = args.freq

    data_sets = []
    for data_path in args.data_path:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len],
            target=args.target,
        )
        data_sets.append(data_set)

    data_sets = ConcatDataset(data_sets)
    print(flag, len(data_sets))
    data_loader = DataLoader(
        data_sets,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_sets, data_loader