import argparse
import os
import random
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_path')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--epsilon', type=float, default=0.0)
parser.add_argument('--targeted', type=bool, default=False)
parser.add_argument('--attack_target', type=str, default='next_class', choices=['next_class', 'furthest_class'])
parser.add_argument('--attack_loss', type=str, default='Default', choices=['Default', 'ModelLoss', 'CrossEntropy'])
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

# For FastFeatureAttack
class_to_indices = {}
for i in range(len(dataset)):
    label = dataset.__getitem__(i, True)
    if label not in class_to_indices:
        class_to_indices[label] = []
    class_to_indices[label].append(i)
# it can happen that some classes do not have test samples, in that case, use the
# previous class
for k in range(num_classes):
    if k in class_to_indices:
        prev_class = class_to_indices[k]
        break
for k in range(num_classes):
    if k not in class_to_indices:
        class_to_indices[k] = prev_class
    else:
        prev_class = class_to_indices[k]

model = torch.load(args.model_path, weights_only=False, map_location=device)
model.eval()

def predict(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
one_off = OneOff()
mae = MeanAbsoluteError()
qwk = QuadraticWeightedKappa()

def cross_entropy_loss(pred, true):
    probs = model.loss.to_proba(pred)
    return F.nll_loss(torch.log(probs), true)

adversary = None

if args.attack in ['GradientSignAttack', 'LinfBasicIterativeAttack', 'PGDAttack', 'MomentumIterativeAttack']:
    if args.attack_loss == 'CrossEntropy':
        adversary = getattr(attacks, args.attack)(model, lambda pred, true: cross_entropy_loss(pred, true), 
                                                  eps=args.epsilon, targeted=args.targeted)
    else:
        adversary = getattr(attacks, args.attack)(model, lambda pred, true: model.loss(pred, true).sum(), 
                                            eps=args.epsilon, targeted=args.targeted)


elif args.attack == 'FFA':
    adversary = attacks.FastFeatureAttack(predict, eps=args.epsilon)
        
for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)

    if adversary:
        if args.attack in ['GradientSignAttack', 'LinfBasicIterativeAttack', 'PGDAttack', 
                           'MomentumIterativeAttack']:
            if args.targeted:
                if args.attack_target == 'next_class':
                    target_labels = torch.where(labels == num_classes - 1, labels - 1, labels + 1)
                elif args.attack_target == 'furthest_class':
                    target_labels = torch.where(labels < num_classes // 2, num_classes - 1, 0)
                images_perturbed = adversary.perturb(images, target_labels)
            else:
                images_perturbed = adversary.perturb(images, labels)
        elif args.attack == 'FFA':
            if args.targeted:
                if args.attack_target == 'next_class':
                    target_labels = torch.where(labels == num_classes - 1, labels - 1, labels + 1)
                elif args.attack_target == 'furthest_class':
                    target_labels = torch.where(labels < num_classes // 2, num_classes - 1, 0)
                
                guide_images = []
                for target_label in target_labels:
                    target_idx = random.choice(class_to_indices[target_label.item()])
                    guide_img, _ = dataset[target_idx]
                    guide_images.append(guide_img)

                guide_images = torch.stack(guide_images).to(device)
                        
                images_perturbed = adversary.perturb(images, guide_images)
    else:
        images_perturbed = images

    with torch.no_grad():
        preds = model(images_perturbed)
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

if args.attack == "GradientSignAttack":
    attack = "GSA"
elif args.attack == "LinfBasicIterativeAttack":
    attack = "LBIA"
elif args.attack == "PGDAttack":
    attack = "PGDA"
elif args.attack == "MomentumIterativeAttack":
    attack = "MIA"
elif args.attack == "FFA":
    attack = "FFA"

if args.attack_loss == 'Default':
    if args.attack == 'FFA':
        attack_loss = "MeanSquaredError"
else:
    attack_loss = args.attack_loss

if args.attack == 'none':
    attack = 'None'
    attack_loss = 'None'

print(f"{attack},{attack_loss},{args.dataset},{loss},{args.epsilon},{targeted},{target},{results['Accuracy']},{results['OneOffAccuracy']},{results['MAE']},{results['QWK']}")
