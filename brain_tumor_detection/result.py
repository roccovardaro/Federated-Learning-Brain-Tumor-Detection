import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(accuracy: list, loss :list, num_rounds:int):

    rounds= list(range(0, num_rounds))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Grafico della Loss
    ax1.plot(rounds, loss, 'r', label='Training Loss', linewidth=2, marker='o', markersize=5)
    ax1.set_xlabel('round', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(rounds, accuracy, 'b', label='Training Accuracy', linewidth=2, marker='o', markersize=5)
    ax2.set_ylabel('Accuracy', fontsize=14, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, 1)

    ax1.set_xticks(rounds)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    plt.title('Training Loss and Accuracy', fontsize=16)
    fig.tight_layout()

    plt.show()
