from torch import optim
#from gcn import GNNStack, test

from datasets.gcn.train_test_splitter import TrainTestSplitter
from datasets.gcn.subset import convert_subsets_to_loaders


def train(dataset, train_date, test_date, batch_size=1):
    splitter = TrainTestSplitter(dataset)
    train_set, test_set = splitter.get_train_test_split(train_date, test_date)
    train_sets, valid_sets = splitter.get_train_valid_split(train_set)

    train_loaders = convert_subsets_to_loaders(train_sets)
    valid_loaders = convert_subsets_to_loaders(valid_sets)
    test_loaders = convert_subsets_to_loaders(test_set)

    # build model
    model = GNNStack(dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
        total_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            loss = model.loss(pred.ravel(), label.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        print("Train Loss", total_loss, epoch)

        if epoch % 1 == 0:
            valid_acc = test(model, valid_loader)
            print("Epoch {}. Loss: {:.4f}. Validation accuracy: {:.4f}".format(
                epoch, total_loss, valid_acc))
    return model