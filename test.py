import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_path')
args = parser.parse_args()

import torch
import data
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = getattr(data, args.dataset)('/data/ordinal', transforms)
num_classes = dataset.K
dataloader = torch.utils.data.DataLoader(dataset, 32, False, num_workers=4, pin_memory=True)

model = torch.load(args.model_path, weights_only=False)
model.to(device)
model.eval()

accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
one_off = 0
mae_sum = 0

total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        preds = preds.argmax(dim=1)

        accuracy.update(preds, labels)
        one_off += ((preds == labels) | (torch.abs(preds - labels) == 1)).sum().item()
        mae_sum += torch.abs(preds - labels).sum().item()

        total += labels.size(0)

print(f'Accuracy: {accuracy.compute().item():.4f}')
print(f'One-off Accuracy: {one_off / total:.4f}')
print(f'Mean Absolute Error (MAE): {mae_sum / total:.4f}')