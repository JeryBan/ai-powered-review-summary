
from typing import Dict, List
import matplotlib.pyplot as plt

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    
    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    f1_score = results['f1_score']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    # f1 score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_score, label='f1_score')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.legend();
