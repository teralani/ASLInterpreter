def main():
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

    # Build adjacency using detected joint count
    adj = build_skeleton_adjacency(num_joints=train_set.num_joints)

    num_classes = train_set.num_classes

    model = STPoseModel(
        adj=adj,
        num_classes=num_classes,
        hidden_dim=128,
        num_heads=4,
        T=30
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    # DataLoader settings (Windows-friendly; pin_memory only for CUDA)
    pin_memory = True if device == "cuda" else False

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            # guard NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered NaN/Inf loss; skipping step")
                continue

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # Final Test Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total if total > 0 else 0.0
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()