from unet import *
from dataloader import *

def train(model, criterion, optimzer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eavl()

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, targets in neuron_dataloader[phase]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimzer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimzer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += IoU(outputs, targets)
