# scripts/train_smoke_detr.py
import os, json, torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs/checkpoints", exist_ok=True)

# 간단 입력(무작위 이미지 2장)
def make_dummy(size=(224, 224)):
    import numpy as np
    arr = (np.random.rand(*size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)

images = [make_dummy(), make_dummy()]

# 라벨도 최소 형태(박스 1개, 클래스 1)
def dummy_labels():
    boxes = [[0.2, 0.2, 0.4, 0.4]]  # cx, cy, w, h (0~1)
    labels = [1]
    return {"boxes": torch.tensor(boxes), "class_labels": torch.tensor(labels)}

labels = [dummy_labels(), dummy_labels()]

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

enc = processor(images=images, annotations=labels, return_tensors="pt")
x = enc["pixel_values"].to(device)
y = [{k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in t.items()} for t in enc["labels"]]

optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
steps = int(os.environ.get("SMOKE_STEPS", "10"))
loss_log = []

model.train()
for i in range(steps):
    out = model(pixel_values=x, labels=y)
    loss = sum(out.loss_dict.values())
    optim.zero_grad(); loss.backward(); optim.step()
    loss_log.append(float(loss.item()))
    if (i+1) % 2 == 0:
        print(f"[SMOKE] step {i+1}/{steps} loss={loss.item():.3f}")

# 저장
torch.save(model.state_dict(), "outputs/checkpoints/last-smoke.pt")
with open("outputs/metrics_smoke.json", "w") as f:
    json.dump({"mean_loss": sum(loss_log)/len(loss_log)}, f)
print("[DONE] outputs/checkpoints/last-smoke.pt, outputs/metrics_smoke.json")
