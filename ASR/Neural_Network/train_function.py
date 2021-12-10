import torch
import torch.nn.functional as F
from Utils.viewing import decoder
from Utils.utils_dataset import int_list_to_ipa
from Utils.utils_dataset import int_list_to_char

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, xs, ys) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        xs, ys = xs.to(device), ys.to(device)

        # Compute prediction error
        pred = model(X)
        pred = F.log_softmax(pred, -1)
        pred = torch.transpose(pred, 0, 1)
        loss = loss_fn(pred, y, xs-13, ys)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            pred = pred.transpose(0, 1)
            pred = pred[0]
            pred = decoder(pred)
            print(pred)
            print(int_list_to_char(y[0][:ys[0]].cpu().numpy()))
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
