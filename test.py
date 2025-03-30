import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_path')
parser.add_argument('output')
parser.add_argument('--attack', type=str, default=None)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--targeted', type=bool, default=False)
args = parser.parse_args()

import torch
import data
from mymetrics import OneOff, MeanAbsoluteError 
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import v2
from advertorch import attacks

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    #v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = getattr(data, args.dataset)('/data/ordinal', transforms)
num_classes = dataset.K
dataloader = torch.utils.data.DataLoader(dataset, 32, False, num_workers=4, pin_memory=True)

model = torch.load(args.model_path, weights_only=False, map_location=device)
model.eval()

accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
one_off = OneOff()
mae = MeanAbsoluteError()

adversary = None
if args.attack == 'GradientSignAttack':
    adversary = attacks.GradientSignAttack(model, lambda pred, true: model.loss(pred, true).sum(), 
                                           eps=args.epsilon, targeted = args.targeted)

for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)

    if adversary:
        if adversary.targeted:
            target_labels = (labels + 1) % num_classes
        else:
            target_labels = labels
            images.requires_grad = True
            images = adversary.perturb(images, target_labels)

    with torch.no_grad():
        preds = model(images)
    preds = model.loss.to_classes(preds)

    accuracy.update(preds, labels)
    one_off.update(preds, labels)
    mae.update(preds, labels)

results = (
    f'Accuracy: {accuracy.compute().item():.4f}\n'
    f'One-off Accuracy: {one_off.compute():.4f}\n'
    f'Mean Absolute Error (MAE): {mae.compute():.4f}\n'
)

with open(args.output, 'w') as f:
    f.write(results)
