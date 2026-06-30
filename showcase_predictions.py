from pathlib import Path
from datetime import datetime
from typing import List, Dict

import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


def get_coco_images(base_dir: Path, count: int = 5) -> List[Path]:
    img_dir = base_dir / "datasets" / "coco128" / "images" / "train2017"
    images = sorted(img_dir.glob("*.jpg"))
    if len(images) < count:
        raise FileNotFoundError(f"Only found {len(images)} images in {img_dir}, need {count}.")
    return images[:count]


def load_imagenet_classes(base_dir: Path) -> List[str]:
    classes_file = base_dir / "imagenet_classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(
            "imagenet_classes.txt not found. Run resnet18 demo once or place the file in project root."
        )
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_resnet18_predictions(images: List[Path], classes: List[str]) -> Dict[str, List[str]]:
    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    results: Dict[str, List[str]] = {}
    print("\n=== ResNet18 (Image Classification) - Top 5 predictions ===")

    with torch.no_grad():
        for image_path in images:
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)[0]
            probs = torch.nn.functional.softmax(output, dim=0)
            top5_prob, top5_idx = torch.topk(probs, 5)

            lines = []
            print(f"\n{image_path.name}")
            for rank in range(5):
                label = classes[top5_idx[rank].item()]
                conf = top5_prob[rank].item() * 100
                line = f"  {rank + 1}. {label}: {conf:.2f}%"
                print(line)
                lines.append(line)
            results[image_path.name] = lines

    return results


def run_yolo_predictions(images: List[Path], model_path: str, output_dir: Path, title: str) -> Dict[str, List[str]]:
    model = YOLO(model_path)
    results = model([str(p) for p in images], device="cpu", imgsz=640, conf=0.25, verbose=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, List[str]] = {}

    print(f"\n=== {title} (Object Detection) ===")

    for r in results:
        image_name = Path(r.path).name
        plotted = r.plot()
        save_path = output_dir / image_name
        cv2.imwrite(str(save_path), plotted)

        det_lines: List[str] = []
        if r.boxes is None or len(r.boxes) == 0:
            msg = "  No objects detected"
            print(f"\n{image_name}\n{msg}")
            det_lines.append(msg)
            summary[image_name] = det_lines
            continue

        cls_ids = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        names = r.names

        pairs = sorted(
            [(names[int(c)], float(s)) for c, s in zip(cls_ids, confs)],
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"\n{image_name}")
        for idx, (label, score) in enumerate(pairs[:5], start=1):
            line = f"  {idx}. {label}: {score * 100:.2f}%"
            print(line)
            det_lines.append(line)
        summary[image_name] = det_lines

    return summary


def write_summary(out_dir: Path, resnet: Dict[str, List[str]], y8: Dict[str, List[str]], y10: Dict[str, List[str]]) -> None:
    lines = ["MODEL EVOLUTION SHOWCASE SUMMARY", ""]

    lines.append("ResNet18 (classification probabilities)")
    for image, preds in resnet.items():
        lines.append(f"- {image}")
        lines.extend(preds)
    lines.append("")

    lines.append("YOLOv8n (detection confidences)")
    for image, preds in y8.items():
        lines.append(f"- {image}")
        lines.extend(preds)
    lines.append("")

    lines.append("YOLOv10n (detection confidences)")
    for image, preds in y10.items():
        lines.append(f"- {image}")
        lines.extend(preds)

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    images = get_coco_images(base_dir, count=5)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / "runs" / f"showcase_{stamp}"
    y8_dir = out_dir / "yolov8n"
    y10_dir = out_dir / "yolov10n"

    classes = load_imagenet_classes(base_dir)
    resnet_summary = run_resnet18_predictions(images, classes)
    y8_summary = run_yolo_predictions(images, "yolov8n.pt", y8_dir, "YOLOv8n")
    y10_summary = run_yolo_predictions(images, "yolov10n.pt", y10_dir, "YOLOv10n")

    write_summary(out_dir, resnet_summary, y8_summary, y10_summary)

    print("\n=== Done ===")
    print(f"Saved detection images and summary in: {out_dir}")


if __name__ == "__main__":
    main()
