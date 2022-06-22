from gcn.train import train


def main():
    train_dates = [['2002-01-02', '2009-04-16'],
                   ['2002-01-02', '2012-12-06'],
                   ['2002-01-02', '2016-08-01'],
                   ['2002-01-02', '2018-10-19'],
                   ['2002-01-02', '2019-06-23']]
    test_dates = [['2010-03-15', '2010-07-21'],
                  ['2013-11-04', '2014-03-14'],
                  ['2017-06-28', '2017-11-02'],
                  ['2019-06-09', '2019-10-16'],
                  ['2019-11-12', '2020-03-24']]

    # train_loader, test_loader = split_dataset_datewise(train_days[0], test_days[0], root_folder=data_dir)

    model = train(dataset, train_daates[0], test_dates[0])

if __name__ == '__main__':
    main()