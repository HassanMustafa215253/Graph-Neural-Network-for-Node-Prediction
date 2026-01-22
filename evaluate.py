
import argparse
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score

from models.gcn import GCN
from utils.data_loader import load_cora, get_device
from utils.visualization import visualize_embeddings, plot_confusion_matrix, CORA_CLASSES


def load_model(model_path, device):
   
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    dataset, data = load_cora()
    
    model = GCN(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_channels=args.hidden,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, data.to(device)


@torch.no_grad()
def evaluate_model(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Get test set predictions
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "=" * 60)
    print("Evaluation Results on Test Set")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:         {accuracy*100:.2f}%")
    print(f"  F1 Score (Macro): {f1_macro*100:.2f}%")
    print(f"  F1 Score (Weighted): {f1_weighted*100:.2f}%")
    
    print(f"\nPer-Class Classification Report:")
    print("-" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=CORA_CLASSES,
        digits=4
    )
    print(report)
    
    return y_true, y_pred, accuracy


def main(args):
    print("=" * 60)
    print("GCN Model Evaluation")
    print("=" * 60)
    
    device = get_device()
    
    print(f"\nLoading model from {args.model_path}...")
    model, data = load_model(args.model_path, device)
    
    # Run evaluation
    y_true, y_pred, accuracy = evaluate_model(model, data)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_embeddings(model, data, save_path='embeddings_tsne_eval.png')
    plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained GCN model')
    parser.add_argument('--model_path', type=str, default='gcn_cora_model.pt',
                        help='Path to saved model checkpoint')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
