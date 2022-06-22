from datetime import datetime, timedelta


class DateCreator:
    def __init__(self, start_end_date, df_dates, n_window, t_start_delay=10):
        self.t_start_delay = t_start_delay if t_start_delay >= n_window else n_window
        self.__dates = self.create_date_interval(start_end_date, df_dates)
        self.start_date = self.__dates[0]
        self.end_date = self.__dates[-1]

    def create_date_interval(self, start_end_date, df_dates):
        start, end = start_end_date
        start = datetime.strptime(start, '%Y-%m-%d') + timedelta(days=self.t_start_delay)
        end = datetime.strptime(end, '%Y-%m-%d')
        delta = end - start

        interval = [start + timedelta(days=i) for i in range(delta.days + 1)]
        interval = [str(date.date()) for date in interval]
        interval = [date for date in interval if date in df_dates]
        return interval

    def get_index(self, idx):
        if isinstance(idx, list):
            return [self.__dates[i] for i in idx]
        return self.__dates[idx]

    def __getitem__(self, idx):
        return self.__dates[idx]

    def __len__(self):
        return len(self.__dates)