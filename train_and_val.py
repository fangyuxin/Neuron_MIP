from unet import *
from dataloader import *
from metrics import *

def train(model, criterion, optimzer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    running_scores = RunningScore()

    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

        running_loss = 0.0
        running_scores.reset()

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
            m = torch.nn.Softmax2d()
            hard_outputs = hardMax(m(outputs)[:, 1, :, :]).int()
            # print(hard_outputs)
            running_scores.update(hard_outputs.numpy(), targets.numpy())

        epoch_loss = running_loss / dataset_size[phase]
        epoch_score = running_scores.get_scores()['Mean IoU']

        print('{} Loss: {:.4f} IoU: {:.4f}'.format(
            phase, epoch_loss, epoch_score))

        if phase == 'val' and epoch_score > best_score:
            best_score = epoch_score
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_score))

    model.load_state_dict(best_model_wts)
    return model


model = Unet()
model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0]))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


model = train(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=3)













