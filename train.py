import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('output')
# FIXME: was 100, increasing when Plateau halt condition was added (see below)
parser.add_argument('--epochs', type=int, default=10000)
args = parser.parse_args()

import torch
import data
import losses
from torchvision.transforms import v2
import torchvision
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATA ##############################

dataset_module = getattr(data, args.dataset)

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(0.5 if dataset_module.HFLIP else 0),
    v2.RandomVerticalFlip(0.5 if dataset_module.VFLIP else 0),
    v2.ColorJitter(0.1, 0.1),
    v2.ToDtype(torch.float32, True),
    #v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = dataset_module('/data/ordinal', transforms)
num_classes = dataset.K

dataset = data.split(dataset, 'train', 0)
dataset = torch.utils.data.DataLoader(dataset, 32, True, num_workers=4, pin_memory=True)

############################## MODEL ##############################

loss_function = getattr(losses, args.loss)(num_classes)

# FIXME: using resnet18 to see if Binomial improves
#model = torchvision.models.resnet50()#weights='DEFAULT')
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, loss_function.how_many_outputs())
model.loss = loss_function
model.to(device)
# FIXME: using Adam instead of AdamW to see if Binomial improves
#opt = torch.optim.AdamW(model.parameters())
opt = torch.optim.Adam(model.parameters())
# FIXME: trying scheduler Reduce LR on Plauteau to see if Binomial improves
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

############################## TRAIN LOOP ##############################

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
    scheduler.step(avg_loss)
    end_time = time.time()
    print(f'Time: {end_time - start_time:.2f}s, Loss: {avg_loss}')
    lr = [group['lr'] for group in opt.param_groups][0]
    # FIXME: stop condition when loss stops decreasing after three plateaus
    if lr < 1e-5:  # 1e-3 (default) -> 1e-4 -> 1e-5 -> stop
        print('Stop due to small lr:', lr, 'after', epoch, 'epochs')
        break

torch.save(model.cpu(), args.output)
