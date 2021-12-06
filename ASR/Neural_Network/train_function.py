import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, xs, ys) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        xs, ys = xs.to(device), ys.to(device)

        # Compute prediction error
        pred = model(X)
        pred = torch.transpose(pred, 0, -1)
        pred = torch.transpose(pred, 1, -1)
        loss = loss_fn(pred, y, xs-10, ys)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
