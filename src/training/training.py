
import torch
from torch import nn

def train_step(model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim,
               acc_fn,
               dataloader: torch.utils.data,
               device: torch.device):
    '''Training step during model training'''

    model.train()
    model.to(device)

    train_loss, train_acc = [], []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        preds = torch.round(logits)
        
        loss = loss_fn(logits, y)
        acc = acc_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            print(f'train loss: {loss:.4f} train acc: {acc:.2f}%')

            train_loss.append(loss.item())
            train_acc.append(acc.item())


    return train_loss, train_acc



def test_step(model: nn.Module,
              loss_fn: nn.Module,
              acc_fn,
              f1,
              dataloader: torch.utils.data,
              device: torch.device):
    '''Test step during evaluation'''

    model.eval()
    model.to(device)

    test_loss, test_acc, f1_score = [], [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            test_logits = model(X)
            test_preds = torch.round(test_logits)
            
            loss = loss_fn(test_logits, y).item()
            acc = acc_fn(test_preds, y).item()
            score = f1(test_preds, y).item()

            if batch % 15 == 0:
                print(f'test loss: {loss:.4f} test acc: {acc:.2f}%, f1 score: {score:.2f}')

                test_loss.append(loss)
                test_acc.append(acc)
                f1_score.append(score)


    return test_loss, test_acc, f1_score



def train(model: nn.Module,
         loss_fn: nn.Module,
         optimizer: torch.optim,
         train_dataloader: torch.utils.data,
         test_dataloader: torch.utils.data,
         acc_fn,
         f1,
         epochs: int,
         device: torch.device):

    results = {}

    for epoch in range(epochs):
        print(f'\nEpoch: {epoch}\n--------')

        train_loss, train_acc = train_step(model=model, loss_fn=loss_fn, optimizer=optimizer, acc_fn=acc_fn, dataloader=train_dataloader, device=device)
        results['train_loss'] = train_loss
        results['train_acc'] = train_acc

        print()
        
        test_loss, test_acc, f1_score = test_step(model=model, loss_fn=loss_fn, acc_fn=acc_fn, f1=f1, dataloader=test_dataloader, device=device)
        results['test_loss'] = test_loss
        results['test_acc'] = test_acc
        results['f1_score'] = f1_score


    return results
