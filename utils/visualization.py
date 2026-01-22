
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch


# Cora class labels
CORA_CLASSES = [
    'Case_Based',
    'Genetic_Algorithms', 
    'Neural_Networks',
    'Probabilistic_Methods',
    'Reinforcement_Learning',
    'Rule_Learning',
    'Theory'
]


def plot_training_curves(train_losses, train_accs, val_accs, save_path='training_curves.png'):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    epochs = range(1, len(train_accs) + 1)
    axes[1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def visualize_embeddings(model, data, save_path='embeddings_tsne.png'):
    
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
    
    labels = data.y.cpu().numpy()
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    
    plt.figure(figsize=(10, 8))
    
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(CORA_CLASSES)))
    
    for i, class_name in enumerate(CORA_CLASSES):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.7,
            s=20
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.title('Node Embeddings Visualization (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Embeddings visualization saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(CORA_CLASSES))
    plt.xticks(tick_marks, CORA_CLASSES, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, CORA_CLASSES, fontsize=9)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
