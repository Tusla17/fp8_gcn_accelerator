import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.gcn import GCN2Layer
from utils.data_loader import load_planetoid_dataset

def train_epoch(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()
    
    features = data['features'].to(device)
    adj_norm = data['adj_norm'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    
    out = model(features, adj_norm)
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    
    pred = out[train_mask].max(1)[1]
    acc = pred.eq(labels[train_mask]).sum().item() / train_mask.sum().item()
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), acc

@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    features = data['features'].to(device)
    adj_norm = data['adj_norm'].to(device)
    labels = data['labels'].to(device)
    mask = mask.to(device)
    
    out = model(features, adj_norm)
    loss = F.nll_loss(out[mask], labels[mask])
    
    pred = out[mask].max(1)[1]
    acc = pred.eq(labels[mask]).sum().item() / mask.sum().item()
    
    return loss.item(), acc

def run_experiment(dataset_name="cora", hidden_dim=16, lr=0.01, weight_decay=5e-4, 
                   dropout=0.5, max_epochs=200, patience=20, device="cpu", save_dir="software/checkpoints"):
    
    print(f"\n{'='*70}")
    print(f"  [DATASET: {dataset_name.upper()}] Training FP32 Baseline GCN (2 Layers)")
    print(f"{'='*70}")
    
    # 1. Load data
    data = load_planetoid_dataset(dataset_name=dataset_name, root_dir="software/data")
    print(f"  Nodes: {data['num_nodes']} | Features: {data['num_features']} | Classes: {data['num_classes']}")
    print(f"  Train: {data['train_mask'].sum().item()} | Val: {data['val_mask'].sum().item()} | Test: {data['test_mask'].sum().item()}")
    
    # 2. Build model
    model = GCN2Layer(
        in_features=data['num_features'],
        hidden_dim=hidden_dim,
        num_classes=data['num_classes'],
        dropout=dropout,
        bias=False
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_cnt = 0
    best_model_weights = None
    
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_epoch(model, optimizer, data, device)
        val_loss, val_acc = evaluate(model, data, data['val_mask'], device)
        test_loss, test_acc = evaluate(model, data, data['test_mask'], device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_cnt = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            
        if epoch % 20 == 0 or epoch == max_epochs or patience_cnt >= patience:
            print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            
        if patience_cnt >= patience:
            print(f"  --> Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.4f})")
            break
            
    elapsed = time.time() - start_time
    print(f"  Training finished in {elapsed:.2f}s.")
    print(f"  >>> BEST Test Accuracy: {best_test_acc * 100:.2f}% (Val Acc: {best_val_acc * 100:.2f}%)")
    
    # 3. Save weights
    os.makedirs(save_dir, exist_ok=True)
    save_path_pth = os.path.join(save_dir, f"gcn_{dataset_name.lower()}_fp32.pth")
    save_path_npz = os.path.join(save_dir, f"gcn_{dataset_name.lower()}_fp32_weights.npz")
    
    # Save PyTorch state dict
    torch.save({
        'state_dict': best_model_weights,
        'dataset': dataset_name,
        'num_features': data['num_features'],
        'hidden_dim': hidden_dim,
        'num_classes': data['num_classes'],
        'test_acc': best_test_acc,
        'val_acc': best_val_acc
    }, save_path_pth)
    
    # Save numpy weights for C/Verilog/Testvector generator
    w0 = best_model_weights['conv1.weight'].numpy()
    w1 = best_model_weights['conv2.weight'].numpy()
    np.savez(save_path_npz, W0=w0, W1=w1, test_acc=best_test_acc)
    
    print(f"  Saved checkpoint: {save_path_pth}")
    print(f"  Saved NumPy weights: {save_path_npz} (W0 shape: {w0.shape}, W1 shape: {w1.shape})")
    
    return {
        'dataset': dataset_name,
        'test_acc': best_test_acc,
        'val_acc': best_val_acc,
        'nodes': data['num_nodes'],
        'features': data['num_features'],
        'classes': data['num_classes'],
        'w0_shape': w0.shape,
        'w1_shape': w1.shape
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FP32 Baseline GCN on Planetoid Datasets")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"], help="Datasets to train")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension (Paper uses 16)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    results = []
    for d in args.datasets:
        res = run_experiment(
            dataset_name=d,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            max_epochs=args.epochs,
            patience=args.patience,
            device=args.device
        )
        results.append(res)
        
    print(f"\n{'='*70}")
    print("  SUMMARY: FP32 BASELINE GCN BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12} | {'Nodes':<8} | {'Features':<10} | {'Classes':<8} | {'Test Acc (%)':<14} | {'Ref Paper Acc (%)'}")
    print(f"  {'-'*66}")
    ref_accs = {'cora': '81.5%', 'citeseer': '70.3%', 'pubmed': '79.0%'}
    for r in results:
        ref = ref_accs.get(r['dataset'].lower(), 'N/A')
        print(f"  {r['dataset'].upper():<12} | {r['nodes']:<8} | {r['features']:<10} | {r['classes']:<8} | {r['test_acc']*100:<14.2f} | {ref}")
    print(f"{'='*70}\n")
