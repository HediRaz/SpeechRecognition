import torch
import torch.nn.functional as F
from Utils.viewing import greedy_decoder
from Utils.utils_dataset import int_list_to_ipa
from Utils.utils_dataset import int_list_to_char
from Utils.viewing import MetricsPrint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(dataloader, model, loss_fn, optimizer, metrics=[], decoder=None):
    """ Train function """
    # Initialize training
    printer = MetricsPrint(metrics)
    size = len(dataloader.dataset)
    printer.initial_print(size, name=" "*2+"Train batch"+" "*3)
    model.train()

    # Iterate over the dataset
    for batch, (X, y, xs, ys) in enumerate(dataloader):
        # Work with the GPU if available
        X, y = X.to(device), y.to(device)
        xs, ys = xs.to(device), ys.to(device)

        # Compute prediction error
        preds = model(X)
        preds = F.log_softmax(preds, -1)
        preds = torch.transpose(preds, 0, 1)
        loss = loss_fn(preds, y, xs-10, ys)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print metrics
        if batch % 30 == 0:
            preds = preds.transpose(0, 1)
            preds = decoder(preds)
            labels = [int_list_to_ipa(label) for label in y.to("cpu").numpy()]
            metrics_values = [m(preds, labels) for m in metrics]
            loss, current = loss.item(), batch * len(X)
            printer.print_loss_metrics(loss, metrics_values, current)


def test(dataloader, model, loss_fn, metrics=[], decoder=None, scheduler=None):
    """ Test function """
    # Initialize test
    printer = MetricsPrint(metrics)
    size = len(dataloader.dataset)
    printer.initial_print(1, name=" "*6+"Test"+" "*6)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    metrics_values = [0] * len(metrics)

    with torch.no_grad():
        # Iterate over the test dataset
        for batch, (X, y, xs, ys) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            xs, ys = xs.to(device), ys.to(device)

            # Compute loss
            preds = model(X)
            preds = F.log_softmax(preds, -1)
            preds = torch.transpose(preds, 0, 1)
            test_loss += loss_fn(preds, y, xs-10, ys).item()

            # Compute metrics
            preds = torch.transpose(preds, 0, 1)
            preds = decoder(preds)
            labels = [int_list_to_ipa(label) for label in y.to("cpu").numpy()]
            for i in range(len(metrics)):
                metrics_values[i] += metrics[i](preds, labels)

    # Compute mean values
    test_loss /= num_batches
    for i in range(len(metrics)):
        metrics_values[i] /= num_batches

    # Update de learning rate
    if scheduler is not None:
        scheduler.step(test_loss)

    # Print results
    printer.print_loss_metrics(test_loss, metrics_values, 1)

    # Print samples
    print(preds[0])
    print(labels[0])
