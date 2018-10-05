from package import *
#
# class ImgAugTransform:
#     def __init__(self):
#         self.aug = iaa.Sequential([
#             # iaa.Scale((256, 256)),
#             # iaa.Fliplr(0.5),
#             # iaa.PiecewiseAffine(scale=(0.0001, 0.0002)),
#             # iaa.Affine(rotate=(-20, 20), mode='symmetric'),
#             # iaa.Affine(shear=(-20, 20))
#         ])
#
#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)
#
#
# transforms_via_imgaug = ImgAugTransform()
# #
# # # _, label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)
# #
# class ToBinary():
#     def __init__(self, threshold=0, value=255, type = cv2.THRESH_BINARY):
#         self.threshold = threshold
#         self.value = value
#         self.type = type
#
#     def __call__(self, nparray):
#         _, nparray[:, :, 1] = cv2.threshold(nparray[:, :, 1], self.threshold, self.value, self.type)
#         return nparray
#
#
# #
# #
def hardMax(input):
    input = input[1:,...]
    ones = torch.ones(input.size()).long()
    zeros = torch.zeros(input.size()).long()
    return torch.where(input >= 0.5, ones, zeros)


to_binary = hardMax

#
#
#
#
#
#
#
#
#
#
# class ConvertToBinary():
#     def __init__(self, threshold=1, value=255):
#         self.threshold = threshold
#         self.value = value
#
#     def __call__(self, input_tensor):
#         # print(input_tensor.size())
#         # temp_tensor = input_tensor[1, ...]
#         # print('input_tensor.shape: {}'.format(temp_tensor.size()))
#         upper_bound = (torch.ones(input_tensor[1, ...].size()) * self.value).long()
#         lower_bound = torch.zeros(input_tensor[1, ...].size()).long()
#         input_tensor[1, ...] = torch.where(input_tensor[1, ...] >= self.threshold, upper_bound, lower_bound)
#
#         return input_tensor
#
# convert_to_binary = ToBinary()
#
# # a = torch.ones(3, 3, 3) * 10
# # # print(a)
# # print(convert_to_binary(a))
# # print(a[1,...] == a[1])
#
#
data_transform = {
    'train': transforms.Compose([
        # transforms_via_imgaug,
        # convert_to_binary,
        transforms.ToTensor()
    ]),

    'val': transforms.Compose([
        # convert_to_binary,
        transforms.ToTensor()
    ])
}

# # tensor(42)
# # tensor(959)
# # tensor(1221)
# # tensor(33)
# # tensor(1863)
# # tensor(1447)
# # tensor(546)
# # tensor(1729)
# # tensor(2775)
# # tensor(669)
# # tensor(888)

