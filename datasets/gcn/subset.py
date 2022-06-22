from torch.utils.data import Subset
from utils.datecreator import DateCreator
from datasets.gcn.dataset import GraphDataset


class MySubset(Subset):
    def __init__(self, dataset: GraphDataset, start_end_date, t_start_delay=10):
        super(MySubset, self).__init__(dataset, None)
        self.dataset = dataset
        self.dates = DateCreator(start_end_date, self.dataset.all_dates, self.dataset.n, t_start_delay)
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, local_idx):
        if isinstance(local_idx, list):
            dates = [self.dates[i] for i in local_idx]
            global_idxs = [self.dates.index(date) for date in dates]
            return self.dataset[global_idxs]

        date = self.dates[local_idx]
        global_idx = self.dataset.all_dates.index(date)
        return self.dataset[global_idx]


def convert_subsets_to_loaders(subset, batch_size=1, shuffle=False):
    #batch_size = self.batch_size if batch_size is None else batch_size
    if isinstance(subset, list):
        return [DataLoader(s, batch_size=batch_size, shuffle=shuffle) for s in subset]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)