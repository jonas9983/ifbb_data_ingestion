import os
import torch
import argparse
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
import json

def count_categories_from_coco(coco_json_path: str):
    """Read COCO annotations and return number of classes."""
    with open(coco_json_path, "r") as f:
        j = json.load(f)
    cats = j.get("categories", [])
    return len(cats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF-DETR on a custom dataset")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root dataset directory (must contain train/ and valid/)")
    parser.add_argument("--model_variant", type=str, default="medium",
                        choices=["nano", "small", "medium", "large", "base"],
                        help="RF-DETR model variant to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Checkpoint output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (cuda or cpu)")
    args = parser.parse_args()

    # Detect number of classes from validation annotations
    valid_json = os.path.join(args.dataset_dir, "valid", "_annotations.coco.json")
    if not os.path.exists(valid_json):
        raise FileNotFoundError(f"Missing validation annotations at {valid_json}")
    num_classes = count_categories_from_coco(valid_json)
    print(f"Found {num_classes} classes in dataset.")

    # Select the model variant
    if args.model_variant == "nano":
        model = RFDETRNano(num_classes=num_classes)
    elif args.model_variant == "small":
        model = RFDETRSmall(num_classes=num_classes)
    elif args.model_variant == "medium":
        model = RFDETRMedium(num_classes=num_classes)
    elif args.model_variant == "large":
        model = RFDETRLarge(num_classes=num_classes)
    elif args.model_variant == "base":
        model = RFDETRBase(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model variant: {args.model_variant}")

    print(f"Training RF-DETR ({args.model_variant}) on {args.device} for {args.epochs} epochs...")

    # Train the model
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )

    print(f"Training completed. Checkpoints saved in {args.output_dir}")