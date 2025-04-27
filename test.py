import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_path')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--epsilon', type=float, default=0.0)
parser.add_argument('--targeted', type=bool, default=False)
parser.add_argument('--attack_target', type=str, default='next_class', choices=['next_class', 'furthest_class'])
parser.add_argument('--attack_loss', type=str, default='ModelLoss', choices=['ModelLoss', 'CrossEntropy'])
args = parser.parse_args()

import torch
import data
from metrics import OneOff, MeanAbsoluteError, QuadraticWeightedKappa
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import v2
from advertorch import attacks
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
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
    probs = model.loss.to_proba(pred)
    return F.nll_loss(torch.log(probs), true)

adversary = None
if args.attack == 'GSA':
    if (args.attack_loss == 'CrossEntropy'):
        adversary = attacks.GradientSignAttack(model, lambda pred, true: cross_entropy_loss(pred, true), 
                                               eps=args.epsilon, targeted=args.targeted)
    else:
        adversary = attacks.GradientSignAttack(model, lambda pred, true: model.loss(pred, true).sum(), 
                                           eps=args.epsilon, targeted=args.targeted)
elif args.attack == 'FFA':
    if (args.attack_loss == 'CrossEntropy'):
        adversary = attacks.FastFeatureAttack(model, lambda pred, true: cross_entropy_loss(pred, true), 
                                               eps=args.epsilon)
    else:
        adversary = attacks.GradientSignAttack(model, lambda pred, true: model.loss(pred, true).sum(), 
                                           eps=args.epsilon)

for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)

    if adversary:
        images.requires_grad = True
        if args.targeted:
            if args.attack_target == 'next_class':
                target_labels = torch.where(labels == num_classes - 1, labels - 1, labels + 1)
            elif args.attack_target == 'furthest_class':
                target_labels = torch.where(labels < num_classes // 2, num_classes - 1, 0)
            images = adversary.perturb(images, target_labels)
        else:
            images = adversary.perturb(images, labels)

    with torch.no_grad():
        preds = model(images)
    preds = model.loss.to_classes(preds)

    accuracy.update(preds, labels)
    one_off.update(preds, labels)
    mae.update(preds, labels)
    qwk.update(preds, labels)

results = {
    'Accuracy': accuracy.compute().item(),
    'OneOffAccuracy': one_off.compute(),
    'MAE': mae.compute(),
    'QWK': qwk.compute()
}

loss = os.path.basename(args.model_path).split('-')[-1].replace('.pth', '')
targeted = "yes" if args.targeted else "no"
target = args.attack_target if args.targeted else "none"
attack_loss = args.attack_loss if args.attack != 'none' else "none"

print(f"{args.attack},{attack_loss},{args.dataset},{loss},{args.epsilon},{args.targeted},{target},{results['Accuracy']},{results['OneOffAccuracy']},{results['MAE']},{results['QWK']}")

