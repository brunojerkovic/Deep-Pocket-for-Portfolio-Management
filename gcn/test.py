import torch


def test(model, loader, is_validation=False):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in loader:
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()

    # TODO: ovo popravi poslije
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    return correct / total