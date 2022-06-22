from typing import Tuple
from enum import Enum
from datasets.gcn.subset import MySubset


class SplitMethod(Enum):
    TIME_SERIES_SPLIT = 1
    BLOCKED_CV = 2
    MANUAL_SPLIT = 3


class TrainTestSplitter:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset

    def get_train_test_split(self, train_dates, test_dates) -> Tuple[MySubset, MySubset]:
        self.train_dates = train_dates
        self.valid_dates = test_dates
        train_subset = MySubset(self.dataset, train_dates)
        test_subset = MySubset(self.dataset, test_dates)
        return train_subset, test_subset

    def get_train_valid_split(self, train_subset: MySubset, val_ratio=0.1, validation_dates=None,
                              method: SplitMethod = SplitMethod.TIME_SERIES_SPLIT):
        if method == SplitMethod.TIME_SERIES_SPLIT:
            return self.ts_cv_split(train_subset, val_ratio)
        elif method == SplitMethod.BLOCKED_CV:
            return self.blocked_cv_split(train_subset, val_ratio)
        elif method == SplitMethod.MANUAL_SPLIT:
            return self.manual_split(train_subset, validation_dates)
        else:
            raise ValueError("Split method is not available!")

    def ts_cv_split(self, train_subset, val_ratio):
        train_subsets = []
        valid_subsets = []

        fold_size = int(len(train_subset) * val_ratio)
        n_splits = int(1 / val_ratio)
        idxs = [[idx + (i_split * fold_size) for idx in range(fold_size)] for i_split in range(n_splits)]
        all_dates = self.dataset.all_dates

        for i in range(n_splits - 1):
            curr_train_interval = [all_dates[idxs[0][0]], all_dates[idxs[i][-1]]]
            curr_valid_interval = [all_dates[idxs[i + 1][0]], all_dates[idxs[i + 1][-1]]]

            train_subsets.append(MySubset(self.dataset, curr_train_interval))
            valid_subsets.append(MySubset(self.dataset, curr_valid_interval))

        return train_subsets, valid_subsets

    def blocked_cv_split(self, train_subset, val_ratio):
        train_subsets = []
        valid_subsets = []

        fold_size = int(len(train_subset) * val_ratio)
        n_splits = int(1 / val_ratio)
        idxs = [[idx + (i_split * fold_size) for idx in range(fold_size)] for i_split in range(n_splits)]
        all_dates = self.dataset.all_dates

        for i in range(n_splits - 1):
            curr_train_interval = [all_dates[idxs[i][0]], all_dates[idxs[i][-1]]]
            curr_valid_interval = [all_dates[idxs[i + 1][0]], all_dates[idxs[i + 1][-1]]]

            train_subsets.append(MySubset(self.dataset, curr_train_interval))
            valid_subsets.append(MySubset(self.dataset, curr_valid_interval))

        return train_subsets, valid_subsets

    def manual_split(self, train_subset: MySubset, valid_dates: list):
        # Create intervals for training
        train_dates = [train_subset.start_date, train_subset.end_date]
        train_starts = [train_dates[0],
                        *[valid_date[1] for valid_date in valid_dates]]  # The starting of train intervals
        train_ends = [*[valid_date[1] for valid_date in valid_dates], train_dates[1]]  # The ending of train intervals
        train_dates = [[ts, te] for (ts, te) in zip(train_starts, train_ends)]  # Intervals for training

        # Create training/validation subsets
        train_subsets = [MySubset(self.dataset, train_interval) for train_interval in train_dates]
        valid_subsets = [MySubset(self.dataset, valid_interval) for valid_interval in valid_dates]

        # Check if validation interval follows train interval
        if len(train_subsets) == len(valid_subsets) + 1:
            train_subsets = train_subsets[:-1]
        if len(train_subsets) != len(valid_subsets):
            raise ValueError("The number of train and validation date intervals should be the same!")
        return train_subsets, valid_subsets