import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('output')
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

import torch
import data
import losses
from torchvision.transforms import v2
import torchvision
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_module = getattr(data, args.dataset)

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(0.5 if dataset_module.HFLIP else 0),
    v2.RandomVerticalFlip(0.5 if dataset_module.VFLIP else 0),
    v2.ColorJitter(0.1, 0.1),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = dataset_module('/data/ordinal', transforms)
num_classes = dataset.K

dataset = data.split(dataset, 'train', 0)
dataset = torch.utils.data.DataLoader(dataset, 32, True, num_workers=4, pin_memory=True)

loss_function = getattr(losses, args.loss)(num_classes)

model = torchvision.models.resnet50(num_classes=loss_function.how_many_outputs())
model.loss = loss_function
model.to(device)
opt = torch.optim.AdamW(model.parameters())

for epoch in range(args.epochs):
    print(f'Epoch: {epoch+1} / {args.epochs}')
    start_time = time.time()
    avg_loss = 0
    for images, labels in dataset:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        loss = loss_function(preds, labels).mean()
        avg_loss += float(loss) / len(dataset)
        opt.zero_grad()
        loss.backward()
        opt.step()
    end_time = time.time()
    print(f'Time: {end_time - start_time:.2f}s, Loss: {avg_loss}')

torch.save(model, args.output)
