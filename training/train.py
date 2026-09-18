def plot_values(num_epochs, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    epochs = range(0, num_epochs)

    plt.figure(figsize=(8, 5))

    colors = list(mcolors.CSS4_COLORS.keys())
    for i, name, list in enumerate(kwargs):
        plt.plot(epochs, list, marker='o', label=f"{name.capitalize()}", color={colors[i % len(kwargs)]})

    plt.title('Accuracy and Loss graph', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Display the plot
    plt.show()
     
def main():
    import os
    import time
    import argparse
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm


    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.resolve()))

    from model.st_pose_model import STPoseModel
    from training.dataset import ASLDataset
    from utils.skeleton import build_skeleton_adjacency

    device = (
        "xpu" if getattr(torch, "xpu", None) and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="model_files/best_model_w_blanks.pt")
    args = parser.parse_args()

    combined_json = "data/combined_dict.json" 

    print(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(False)

    use_amp = device in ("cuda", "xpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


    train_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json= combined_json,
        split="train"
    )

    val_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json= combined_json,
        split="val"
    )

    test_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json= combined_json,
        split="test"
    )

    print(f"train: {len(train_set)} samples, {train_set.num_classes} classes")
    print(f"val:   {len(val_set)} samples, {val_set.num_classes} classes")
    print(f"test:  {len(test_set)} samples, {test_set.num_classes} classes")

    from collections import Counter
    print("val label distribution:", Counter(val_set.video_to_label.values()).most_common()[-10:])  # rarest classes
    print("test label distribution:", Counter(test_set.video_to_label.values()).most_common()[-10:])

    adj = build_skeleton_adjacency()

    num_classes = train_set.num_classes

    model = STPoseModel(
        adj=adj,
        num_classes=num_classes,
        hidden_dim=256, # 128
        num_heads=8,    # 4
        T=train_set.T,
        dropout=0.2 # before 0.1
    ).to(device)

    if args.eval_only:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    elif input("load from checkpoint (Y/N)? ").strip().lower() == "y":
        state_dict = torch.load('model_files/best_model_combined.pt', weights_only=True)
        model.load_state_dict(state_dict)

    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable, continuing uncompiled: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, # before: 1e-5
        weight_decay=1e-2, # before 1e-4
        foreach=False,
        fused=False
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # before: default label_smoothing

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))

    num_workers = min(8, os.cpu_count() or 0)
    pin_memory = device == "cuda"

    loader_kwargs = dict(
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = num_workers > 0,
        prefetch_factor = 4 if num_workers > 0 else None
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, **loader_kwargs)

    def to_device(x, mask, y):
        return (
            x.to(device, non_blocking = pin_memory),
            mask.to(device, non_blocking = pin_memory),
            y.to(device, non_blocking = pin_memory),
        )

    # Training loop
    if not args.eval_only:
        EPOCHS = 200 # 30
        best_val_acc = 0.0
        best_ckpt_path = "model_files/best_model_w_blanks.pt"

        best_val_loss = 1e2

    for epoch in range(EPOCHS if not args.eval_only else 0):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):

            (x, mask), y, stems = batch
            x, mask, y = to_device(x, mask, y)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, enabled=use_amp, dtype=torch.bfloat16 if device == "xpu" else torch.float16):
                logits = model(x, mask)
                loss = criterion(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Encountered NaN/Inf loss; skipping step. stems: {stems}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                print(f"Skipping optimizer step: non-finite grad norm ({grad_norm})")

            scaler.update()

            train_loss += loss.item()

        train_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                (x, mask), y, _ = batch
                x, mask, y = to_device(x, mask, y)

                with torch.amp.autocast(device_type="cuda", enabled = use_amp):
                    logits = model(x, mask)
                    loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_loss)
        

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            # f"train_correct / train_total: {train_correct / train_total} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # track best val_loss and a no-improvement counter
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 20:
            print(f"Early stopping at epoch {epoch+1}: no val improvement in 20 epochs")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> new best val acc {best_val_acc:.4f}, saved to {best_ckpt_path}")
    
    if not args.eval_only and os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best checkpoint (val acc {best_val_acc:.4f}) for final test evaluation")

    # final test Evaluation
    model.eval()
    correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            (x, mask), y, _ = batch
            x, mask, y = to_device(x, mask, y)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x, mask)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            top5_preds = logits.topk(min(5, logits.size(1)), dim=1).indices
            top5_correct += top5_preds.eq(y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    test_acc = correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 else 0.0
    print(f"\nFinal Test Top-1 Accuracy: {test_acc:.4f}")
    print(f"Final Test Top-5 Accuracy: {top5_acc:.4f}")


if __name__ == "__main__":
    main()