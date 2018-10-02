from package import *

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Scale((256, 256)),
            iaa.Fliplr(0.5),
            # iaa.PiecewiseAffine(scale=(0.0001, 0.0002)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Affine(shear=(-20, 20))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


transforms_via_imgaug = ImgAugTransform()

# _, label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)

class ConvertToBinary():
    def __init__(self, threshold=0, value=255, type = cv2.THRESH_BINARY):
        self.threshold = threshold
        self.value = value
        self.type = type

    def __call__(self, nparray):
        _, nparray[:, :, 1:] = cv2.threshold(nparray[:, :, 1:], self.threshold, self.value, self.type)
        return nparray

convert_to_binary = ConvertToBinary()



data_transform = {
    'train': transforms.Compose([
        transforms_via_imgaug,
        convert_to_binary,
        transforms.ToTensor()
    ]),

    'val': transforms.Compose([
        transforms.ToTensor()
    ])
}





