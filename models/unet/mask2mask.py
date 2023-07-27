import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cls_model.resnet import ResnetCls
from .layers import unetConv2, unetUp, unetUp_origin
from .init_weights import init_weights
from .UNet import UNet
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import torchvision


class EdgeFilter(nn.Module):
    def __init__(self):
        super(EdgeFilter, self).__init__()

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, filter_size),
            padding=(0, filter_size // 2),
        )
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(filter_size, 1),
            padding=(filter_size // 2, 0),
        )
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        all_filters = np.stack(
            [
                filter_0,
                filter_45,
                filter_90,
                filter_135,
                filter_180,
                filter_225,
                filter_270,
                filter_315,
            ]
        )

        self.directional_filter = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=filter_0.shape,
            padding=filter_0.shape[-1] // 2,
        )
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(
            torch.from_numpy(np.zeros(shape=(all_filters.shape[0],)))
        )

    def forward(self, img):
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)

        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = torch.atan2(
            grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b
        ) * (180.0 / 3.14159)
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        print(height, width)
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(
            1, height, width
        )

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(
            1, height, width
        )

        channel_select_filtered = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative]
        )

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        thin_edges = thin_edges / torch.max(torch.max(thin_edges, dim=2), 3)

        return thin_edges


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        """
        Convert image to its gray one.
        """
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class Mask2Mask(nn.Module):
    def __init__(self, main_model):
        super(Mask2Mask, self).__init__()
        self.main_model = main_model
        # for param in self.main_model.parameters():
        #     param.requires_grad = False
        self.additive_model = UNet(n_channels=2, n_classes=1)
        self.cls_model = ResnetCls()
        self.edge_model = GradLayer()
        # self.gray_transform = torchvision.transforms.Grayscale(1)
        self.freeze_main_model()

    def freeze_main_model(self):
        for param in self.main_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        cls_result = self.cls_model(x)
        output = self.main_model(x)
        # cls_result = (cls_result > 0.5).float()
        # print(cls_result)
        # print(output.shape)
        # new_input = torch.concat((x, output), dim=1)
        # print(output.shape)
        # gray_img = self.gray_transform(x)
        edge = self.edge_model(x)
        new_input = torch.concat((edge, output), dim=1)
        output = self.additive_model(new_input)
        # print(cls_result.shape)
        # print(torch.dot(cls_result.unsqueeze(0), output.view(len(cls_result), -1)).shape)

        # output = self.dotProduct(output, cls_result.unsqueeze(1))

        return output, cls_result

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final


class Mask2MaskGray(nn.Module):
    def __init__(self, main_model):
        super(Mask2MaskGray, self).__init__()
        self.main_model = main_model
        # for param in self.main_model.parameters():
        #     param.requires_grad = False
        self.additive_model = UNet(n_channels=3, n_classes=1)
        self.cls_model = ResnetCls()
        self.edge_model = GradLayer()
        self.gray_transform = torchvision.transforms.Grayscale(1)
        self.freeze_main_model()

    def freeze_main_model(self):
        for param in self.main_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        cls_result = self.cls_model(x)
        output = self.main_model(x)
        # cls_result = (cls_result > 0.5).float()
        # print(cls_result)
        # print(output.shape)
        # new_input = torch.concat((x, output), dim=1)
        # print(output.shape)
        gray_img = self.gray_transform(x)
        edge = self.edge_model(x)
        new_input = torch.concat((edge, output, gray_img), dim=1)
        output = self.additive_model(new_input)
        # print(cls_result.shape)
        # print(torch.dot(cls_result.unsqueeze(0), output.view(len(cls_result), -1)).shape)

        # output = self.dotProduct(output, cls_result.unsqueeze(1))

        return output, cls_result
