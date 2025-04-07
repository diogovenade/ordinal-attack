import argparse
import csv
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
from mymetrics import OneOff, MeanAbsoluteError, QuadraticWeightedKappa
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import v2
from advertorch import attacks
from losses import CrossEntropy
import torch.nn.functional as F

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
qwk = QuadraticWeightedKappa()

def cross_entropy_loss(pred, true):
    # pytorch CrossEntropy works with logits, but we want to give probabilities
    probs = model.to_probabilities(pred)
    return F.nll_loss(torch.log(probs), true)
def model_loss(pred, true):
    return model.compute_loss(pred, true)

adversary = None
if args.attack == 'GradientSignAttack':
    ce_loss = CrossEntropy(num_classes)
    adversary = attacks.GradientSignAttack(model, cross_entropy_loss, eps=args.epsilon, targeted=args.targeted)

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
    qwk.update(preds, labels)

results = {
    'Accuracy': accuracy.compute().item(),
    'One-off Accuracy': one_off.compute(),
    'Mean Absolute Error (MAE)': mae.compute(),
    'Quadratic Weighted Kappa (QWK)': qwk.compute()
}

with open(args.output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)
