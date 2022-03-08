# import numpy as np
# import torch
# from medpy import metric
# from scipy.ndimage import zoom
# import torch.nn as nn
# import cv2
# import SimpleITK as sitk
#
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import matplotlib.pylab as pl
# from matplotlib.colors import ListedColormap
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         # own
#         #target = self._one_hot_encoder(target)
#         target=target[:,:,:,0]+target[:,:,:,1]+target[:,:,:,2]
#         inputs=inputs[:,0,:,:,]+inputs[:,1,:,:,]
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes
#
#
# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum()>0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum()==0:
#         return 1, 0
#     else:
#         return 0, 0
#
#
# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             # our
#             input = torch.from_numpy(slice).unsqueeze(0).float().to(device)
#             # origin
#             # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 # our
#                 out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
#                 # origin
#                 #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#         prediction=prediction[0]
#     else:
#         # our1
#         #input = torch.from_numpy(image).unsqueeze(0).float().to(device)
#         # our2
#         input=torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
#         # origin
#         #input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
#
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     print("jj")
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#     # origin
#     # if test_save_path is not None:
#     #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     #     img_itk.SetSpacing((1, 1, z_spacing))
#     #     prd_itk.SetSpacing((1, 1, z_spacing))
#     #     lab_itk.SetSpacing((1, 1, z_spacing))
#     #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#     #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#     #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     # our
#     if test_save_path is not None:
#         cv2.imwrite(test_save_path + "/" +case +".png",prediction*255)
#     return metric_list
#
# def visualize(X, y, y_pred, sample_num, figsize=(10, 10), cmap='viridis'):
#     y_pred_np = y_pred[sample_num, :, :, :]
#     y_class = np.argmax(y_pred_np, axis=-1)
#     x = X.numpy()[sample_num, :, :, :]
#     y_np = y.numpy()[sample_num, :, :, :]
#     y_np = np.argmax(y_np, axis=-1)
#     fig, axis = plt.subplots(1, 2, figsize=figsize)
#     axis[0].imshow(x, cmap='gray')
#     axis[0].imshow(y_np, cmap=cmap, alpha=0.3)
#     axis[0].set_title("original labels")
#     axis[1].imshow(x, cmap='gray')
#     axis[1].imshow(y_class, cmap=cmap, alpha=0.3)
#     axis[1].set_title("predicted labels")
#     plt.show()
#
#
# def visualize_non_empty_predictions(X, y, models, figsize=(10, 10), cmap=pl.cm.tab10_r, alpha=0.8, titles=[]):
#     x = X.numpy()
#     y_np = y.numpy()
#     y_np = np.argmax(y_np, axis=-1)
#     labels = np.unique(y_np)
#     if len(labels) != 1:
#         # create cmap
#         my_cmap = cmap(np.arange(cmap.N))
#         my_cmap[:, -1] = 0.9
#         my_cmap[0, -1] = 0.1
#         my_cmap = ListedColormap(my_cmap)
#
#         n_plots = len(models) + 1
#         fig, axis = plt.subplots(1, n_plots, figsize=figsize)
#
#         axis[0].imshow(x, cmap='gray')
#         axis[0].imshow(y_np, cmap=my_cmap, alpha=alpha)
#         axis[0].set_title("original labels")
#         axis[0].set_xticks([])
#         axis[0].set_yticks([])
#
#         for i, model in enumerate(models):
#             y_pred = model.model.predict(tf.expand_dims(X, axis=0))
#             y_class = np.argmax(y_pred, axis=-1)
#             axis[i+1].imshow(x, cmap='gray')
#             axis[i+1].imshow(y_class[0], cmap=my_cmap, alpha=alpha)
#             if titles == []:
#                 axis[i+1].set_title(f"{model.name}")
#             else:
#                 axis[i+1].set_title(f"{titles[i]}")
#             axis[i+1].set_xticks([])
#             axis[i+1].set_yticks([])
#
#         plt.show()
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        # our
        gt=gt[:,:,1]
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(image)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        prediction = prediction[0]
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    if test_save_path is not None:
        print(test_save_path + '/'+case + '.png')
        cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list
