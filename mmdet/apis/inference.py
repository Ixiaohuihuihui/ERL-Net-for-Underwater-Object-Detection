import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
#--------new add by hui-----------
import cv2
import numpy as np
from torch.autograd import Function
import torch.nn as nn


class CAM():
    def __init__(self, model):
        self.gradient = []
        self.model = model
        self.h = self.model.model.module.layer[-2].register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        # print("Gradient saved!!!!")
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])
        # print(self.gradient[0].size())

    def get_gradient(self):
        return self.gradient[0]

    def remove_hook(self):
        self.h.remove()

    def normalize_cam(self, x):
        x = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        # x[x<torch.max(x)]=-1
        return x

    def visualize(self, cam_img, guided_img, img_var):
        guided_img = guided_img.numpy()
        cam_img = resize(cam_img.cpu().data.numpy(), output_shape=(28, 28))
        x = img_var[0, :, :].cpu().data.numpy()

        fig = plt.figure(figsize=(20, 12))

        plt.subplot(1, 4, 1)
        plt.title("Original Image")
        plt.imshow(x, cmap="gray")

        plt.subplot(1, 4, 2)
        plt.title("Class Activation Map")
        plt.imshow(cam_img)

        plt.subplot(1, 4, 3)
        plt.title("Guided Backpropagation")
        plt.imshow(guided_img, cmap='gray')

        plt.subplot(1, 4, 4)
        plt.title("Guided x CAM")
        plt.imshow(guided_img * cam_img, cmap="gray")
        plt.show()

    def get_cam(self, idx):
        grad = self.get_gradient()
        alpha = torch.sum(grad, dim=3, keepdim=True)
        alpha = torch.sum(alpha, dim=2, keepdim=True)

        cam = alpha[idx] * grad[idx]
        cam = torch.sum(cam, dim=0)
        cam = self.normalize_cam(cam)
        self.remove_hook()

        return cam

class GuidedBackpropRelu(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        grad_input[input < 0] = 0
        return grad_input


class GuidedReluModel(nn.Module):
    def __init__(self, model, to_be_replaced, replace_to):
        super(GuidedReluModel, self).__init__()
        self.model = model
        self.to_be_replaced = to_be_replaced
        self.replace_to = replace_to
        self.layers = []
        self.output = []

        for m in self.model.modules():
            if isinstance(m, self.to_be_replaced):
                self.layers.append(self.replace_to)
                # self.layers.append(m)
            elif isinstance(m, nn.Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                self.layers.append(m)
            elif isinstance(m, nn.Linear):
                self.layers.append(m)
            elif isinstance(m, nn.AvgPool2d):
                self.layers.append(m)

        for i in self.layers:
            print(i)

    def reset_output(self):
        self.output = []

    def hook(self, grad):
        out = grad[:, 0, :, :].cpu().data  # .numpy()
        print("out_size:", out.size())
        self.output.append(out)

    def get_visual(self, idx, original_img):
        grad = self.output[0][idx]
        return grad

    def forward(self, x):
        out = x
        out.register_hook(self.hook)
        for i in self.layers[:-3]:
            out = i(out)
        out = out.view(out.size()[0], -1)
        for j in self.layers[-3:]:
            out = j(out)
        return out

class ShowGradCam:
    def __init__(self,conv_layer):
        assert isinstance(conv_layer,torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self,module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        Based on gradient and feature map, generate cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self,input_img):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img,str):
            input_img = cv2.imread(input_img)
        img_size = (input_img.shape[1],input_img.shape[0])
        fmap = self.feature_res[0][0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0][0].cpu().data.numpy().squeeze()
        cam = self.gen_cam(fmap, grads_val)
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)/255.
        cam = heatmap + np.float32(input_img/255.)
        cam = cam / np.max(cam)*255
        cv2.imwrite('grad_feature.jpg',cam)
        print('save gradcam result in grad_feature.jpg')

def comp_class_vec(output_vec, index=None):
    """
    :param ouput_vec: tensor
    :param index: int
    :return: tensor
    """
    if not index:
        index = np.argmax(output_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output_vec)  # one_hot = 11.8605

    return class_vec

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    # grad_cam = ShowGradCam(model.neck)
    guided_relu = GuidedBackpropRelu.apply
    guide = GuidedReluModel(model, nn.ReLU, guided_relu)
    cam = CAM(guide)
    guide.reset_output()

    # forward the model
    with torch.no_grad():
        result = model(return_loss=True, rescale=True, **data)
        # grad_cam.show_on_img(data['img'][0][0])
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()
