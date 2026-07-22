def main():
    import os
    import time
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

    print(f"Using device: {device}")

    use_amp = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


    train_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json="data/WLASL_v0.3.json",
        split="train"
    )

    val_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json="data/WLASL_v0.3.json",
        split="val"
    )

    test_set = ASLDataset(
        processed_dir="data/processed",
        wsasl_json="data/WLASL_v0.3.json",
        split="test"
    )

    # Build adjacency matrix
    adj = build_skeleton_adjacency()

    num_classes = train_set.num_classes

    model = STPoseModel(
        adj=adj,
        num_classes=num_classes,
        hidden_dim=128,
        num_heads=4,
        T=train_set.T
    ).to(device)

    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable, continuing uncompiled: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

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

    EPOCHS = 30
    best_val_acc = 0.0
    best_ckpt_path = "model_files/best_mode.pt"

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            (x, mask), y = batch
            x, mask, y = to_device(x, mask, y)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled = use_amp):
                logits = model(x, mask)
                loss = criterion(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered NaN/Inf loss; skipping step")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
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
                (x, mask), y = batch
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
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> new best val acc {best_val_acc:.4f}, saved to {best_ckpt_path}")
    
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best checkpoint (val acc {best_val_acc:.4f}) for final test evaluation")

    # Final Test Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            (x, mask), y = batch
            x, mask, y = to_device(x, mask, y)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x, mask)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total if total > 0 else 0.0
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()