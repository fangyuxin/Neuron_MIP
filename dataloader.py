from data_augmentation import *

class NeuronDataset(Dataset):
    def __init__(self, data_dir, phase, transform=None):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform


    def __len__(self):
        if self.phase == 'train':
            return 20
        if self.phase == 'val':
            return 10

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, '{}'.format(self.phase),
                                       'image', '{}.png'.format(index + 1)))

        label = Image.open(os.path.join(self.data_dir, '{}'.format(self.phase),
                                       'label', '{}.png'.format(index + 1)))

        # label.show()
        #
        # TF.resize(label, (256, 256), interpolation=2).show()

        image = np.array(TF.resize(image, (256, 256), interpolation=2))
        label = np.array(TF.resize(label, (256, 256), interpolation=2))


        image_white_black = np.stack([image, label, label], axis=-1)
        # print(image_white_black[:, :, 1:].shape)

        if self.transform:
            image_white_black = self.transform[self.phase](image_white_black)
        # else:
        #     image_white_black = transforms.ToTensor()

        image = image_white_black[0].view(-1, *image_white_black[0].size())

        # _, label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)
        # label = np.array(label)

        label = image_white_black[1]
        # label[1] = 255 - label[1]

        return image, label


# neuron_dataset = NeuronDataset(DATA_DIR, 'train', data_transform)
#
#
# plt.figure()
# for i in range(1):
#     image_sample, label_sample = neuron_dataset[i]
#     print()
#
#     plt.subplot(1, 3, 1)
#     image = image_sample[0, :, :].numpy()
#     plt.imshow(image, cmap='gray')
#
#     plt.subplot(1, 3, 2)
#     white = label_sample[0, :, :].numpy()
#     plt.imshow(white, cmap='gray')
#
#     plt.subplot(1, 3, 3)
#     black = label_sample[1, :, :].numpy()
#     plt.imshow(black, cmap='gray')
#
#     plt.show()

#
# DATA_DIR = './data'
# neuron_dataset = NeuronDataset(DATA_DIR, 'train', data_transform)
#
#
# image_sample, _ = neuron_dataset[1]
#
# model = Unet()
#
# output = model(image_sample)
#


DATA_DIR = './data'

neuron_dataset = {phase: NeuronDataset(DATA_DIR, phase, data_transform)
                  for phase in ['train', 'val']}

dataset_size = {phase: len(neuron_dataset[phase])
                for phase in ['train', 'val']}

neuron_dataloader = {phase: DataLoader(neuron_dataset[phase], batch_size=2, shuffle=True, num_workers=4)
                     for phase in ['train', 'val']}

# for i, sample in enumerate(neuron_dataloader['train']):
#     model = Unet()
#     output = model(sample[0])
#
#     if i == 2:
#         break


