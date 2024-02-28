
import torch
from torch import nn

def train(model: nn.Module,
         loss_fn: nn.Module,
         optimizer: torch.optim,
         train_dataloader: torch.utils.data,
         test_dataloader: torch.utils.data,
         acc_fn,
         f1,
         epochs: int,
         device: torch.device):

    results = {
        'train_loss' : [],
        'train_acc' : [],
        'test_loss' : [],
        'test_acc' : [],
        'f1_score' : []
    }

    for epoch in range(epochs):
        print(f'\nEpoch: {epoch}\n--------')

        train_loss, train_acc = train_step(model=model, loss_fn=loss_fn, optimizer=optimizer, acc_fn=acc_fn, dataloader=train_dataloader, device=device)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)

        print()
        
        test_loss, test_acc, f1_score = test_step(model=model, loss_fn=loss_fn, acc_fn=acc_fn, f1=f1, dataloader=test_dataloader, device=device)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['f1_score'].append(f1_score)

        print()
        print(f'train loss: {train_loss:.4f} | test loss: {test_loss:.4f}')
        print(f'train acc: {train_acc:.2f}% | test acc: {test_acc:.2f}%')
        print(f'f1 score: {f1_score:.4f}')


    return results


def train_step(model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim,
               acc_fn,
               dataloader: torch.utils.data,
               device: torch.device):

    model.train()
    model.to(device)

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        loss = loss_fn(logits, y)
        acc = acc_fn(preds, y)

        train_loss += loss.item()
        train_acc += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            print(f'train loss: {loss:.4f} train acc: {acc:.2f}%')
            

    train_loss /= len(dataloader)
    train_acc = (train_acc / len(dataloader)) * 100

    return train_loss, train_acc


def test_step(model: nn.Module,
              loss_fn: nn.Module,
              acc_fn,
              f1,
              dataloader: torch.utils.data,
              device: torch.device):

    model.eval()
    model.to(device)

    test_loss, test_acc, f1_score = 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            test_logits = model(X)
            test_preds = torch.argmax(test_logits, dim=1)
            
            test_loss += loss_fn(test_logits, y).item()
            test_acc += acc_fn(test_preds, y).item()
            f1_score += f1(test_preds, y).item()

            if batch % 30 == 0:
                print(f'test loss: {test_loss:.4f} test acc: {test_acc:.2f}%, f1 score: {f1_score:.4f}')

        test_loss /= len(dataloader)
        test_acc = (test_acc / len(dataloader)) * 100

    return test_loss, test_acc, f1_score