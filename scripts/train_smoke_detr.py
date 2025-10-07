# scripts/train_smoke_detr.py
import os, json, torch
from torch import optim
from transformers import DetrForObjectDetection

os.makedirs("outputs/checkpoints", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 불러오기 (가벼운 DETR)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
opt = optim.AdamW(model.parameters(), lr=5e-5)

# 더미 데이터로 짧은 학습
losses = []
for step in range(5):
    x = torch.rand(2, 3, 384, 384, device=device)
    y = [{"boxes": torch.rand(2, 4, device=device),
          "class_labels": torch.randint(0, 91, (2,), device=device)} for _ in range(2)]
    out = model(pixel_values=x, labels=y)
    loss = sum(out.loss_dict.values())
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

# 결과 저장
torch.save(model.state_dict(), "outputs/checkpoints/last-smoke.pt")
json.dump({"mean_loss": sum(losses)/len(losses)}, open("outputs/metrics_smoke.json", "w"))

print("[DONE] outputs/checkpoints/last-smoke.pt")
