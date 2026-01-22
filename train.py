
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gcn import GCN
from utils.data_loader import load_cora, get_device
from utils.visualization import plot_training_curves, visualize_embeddings, plot_confusion_matrix
from sklearn.metrics import classification_report


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    accs = {}
    for split, mask in [('train', data.train_mask), 
                         ('val', data.val_mask), 
                         ('test', data.test_mask)]:
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        accs[split] = correct / total
    
    return accs


def train(args):
    print("=" * 60)
    print("GCN Node Classification on Cora Dataset")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading Cora dataset...")
    dataset, data = load_cora()
    
    # Get device
    device = get_device()
    data = data.to(device)
    
    # Initialize model
    print(f"\n[2/4] Initializing GCN model...")
    model = GCN(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_channels=args.hidden,
        dropout=args.dropout
    ).to(device)
    print(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f"\n[3/4] Training for {args.epochs} epochs...")
    print("-" * 60)
    
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    best_model_state = None
    
    progress_bar = tqdm(range(1, args.epochs + 1), desc="Training")
    
    for epoch in progress_bar:
        loss = train_epoch(model, data, optimizer)
        accs = evaluate(model, data)
        
        train_losses.append(loss)
        train_accs.append(accs['train'])
        val_accs.append(accs['val'])
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss:.4f}",
            'train': f"{accs['train']:.4f}",
            'val': f"{accs['val']:.4f}"
        })
        
        # Save best model
        if accs['val'] > best_val_acc:
            best_val_acc = accs['val']
            best_model_state = model.state_dict().copy()
        
        # Print every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch:03d}: Loss={loss:.4f}, "
                      f"Train={accs['train']:.4f}, Val={accs['val']:.4f}")
    
    # Load best model and evaluate
    print("\n[4/4] Evaluating best model...")
    print("-" * 60)
    model.load_state_dict(best_model_state)
    final_accs = evaluate(model, data)
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {final_accs['train']*100:.2f}%")
    print(f"  Val Accuracy:   {final_accs['val']*100:.2f}%")
    print(f"  Test Accuracy:  {final_accs['test']*100:.2f}%")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'final_accs': final_accs
    }, 'gcn_cora_model.pt')
    print(f"\nModel saved to gcn_cora_model.pt")
    
    # Generate visualizations and metrics
    print("\nGenerating visualizations and evaluation metrics...")
    
    # Training curves
    plot_training_curves(train_losses, train_accs, val_accs)
    
    # t-SNE embeddings
    visualize_embeddings(model, data)
    
    # Confusion matrix on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    plot_confusion_matrix(y_true, y_pred)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    from utils.visualization import CORA_CLASSES
    report = classification_report(
        y_true, y_pred,
        target_names=CORA_CLASSES,
        digits=4
    )
    print(report)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, final_accs


def parse_args():
    parser = argparse.ArgumentParser(description='Train GCN on Cora')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Hidden layer size (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay / L2 regularization (default: 5e-4)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, accs = train(args)
