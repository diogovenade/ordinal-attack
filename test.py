import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_path')
parser.add_argument('output')
parser.add_argument('--attack', type=str, default=None)
parser.add_argument('--epsilon', type=float, default=0.1)
args = parser.parse_args()

import torch
import data
from mymetrics import OneOff, MeanAbsoluteError, Unimodality 
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import v2
from advertorch import attacks

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
one_off = OneOff()
mae = MeanAbsoluteError()
unimodality = Unimodality()

adversary = None
if args.attack:
    adversary = getattr(attacks, args.attack)(model, lambda pred, true: model.loss(pred, true).sum(), eps=args.epsilon)

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if adversary:
            with torch.enable_grad():
                images.requires_grad = True
                images = adversary.perturb(images, labels)

        preds = model(images)
        preds = preds.argmax(dim=1)

        accuracy.update(preds, labels)
        one_off.update(preds, labels)
        mae.update(preds, labels)
        unimodality.update(preds)

results = (
    f'Accuracy: {accuracy.compute().item():.4f}\n'
    f'One-off Accuracy: {one_off.compute():.4f}\n'
    f'Mean Absolute Error (MAE): {mae.compute():.4f}\n'
    f'Unimodality: {unimodality.compute():.4f}\n'
)

with open(args.output, 'w') as f:
    f.write(results)
