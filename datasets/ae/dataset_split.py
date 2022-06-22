from enum import Enum


class SplitType(Enum):
    PERCENTAGE = "PERCENTAGE"
    DATES = "DATES"


class Splitter:
    def __init__(self, split_args):
        self.percs = split_args[0]
        self.date_intervals = split_args[1]

    def split(self, dataset, split_type: str):
        # Check if user entered correct splitting method
        split_type = split_type.upper()
        if split_type not in [e.value for e in SplitType]:
            raise ValueError("Split method not available!")

        # Split the dataset
        subsets = None
        if SplitType[split_type] == SplitType.PERCENTAGE:
            subsets = self.__split_by_perc(dataset, self.percs)
        elif SplitType[split_type] == SplitType.DATES:
            subsets = self.__split_by_dates(dataset, self.date_intervals)

        return subsets

    def __split_by_dates(self, dataset, date_intervals):
        subsets = []
        for date_start, date_end in zip(date_intervals[::2], date_intervals[1::2]):
            subset = dataset[(dataset.date >= date_start) & (dataset.date <= date_end)]
            subsets.append(subset)
        return subsets

    def __split_by_perc(self, dataset, percentages):
        N = len(dataset)
        N_train_perc, N_valid_perc, N_test_perc = percentages
        N_train, N_valid = int(N * N_train_perc), int(N * (N_valid_perc + N_test_perc))

        # Create subsets
        train_set = dataset[:N_train]
        valid_set = dataset[N_train:N_valid]
        test_set = dataset[N_valid:]

        return [train_set, valid_set, test_set]
