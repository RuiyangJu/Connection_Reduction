import torch
from torch import nn

class transitionlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transitionlayer, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self,x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out

class denseblock_1(nn.Module):
    def __init__(self, in_channels):
        super(denseblock_1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv4, conv5], 1))
        conv6 = self.relu(self.conv6(c5_dense))
        c6_dense = self.relu(torch.cat([conv1, conv3, conv5, conv6], 1))
        conv7 = self.relu(self.conv7(c6_dense))
        c7_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv7], 1))
        conv8 = self.relu(self.conv8(c7_dense))
        c8_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv8], 1))
        conv9 = self.relu(self.conv9(c8_dense))
        c9_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv9], 1))
        conv10 = self.relu(self.conv10(c9_dense))
        c10_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv10], 1))
        return c10_dense

class denseblock_2(nn.Module):
    def __init__(self, in_channels):
        super(denseblock_2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=224, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv4, conv5], 1))
        conv6 = self.relu(self.conv6(c5_dense))
        c6_dense = self.relu(torch.cat([conv1, conv3, conv5, conv6], 1))
        conv7 = self.relu(self.conv7(c6_dense))
        c7_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv7], 1))
        conv8 = self.relu(self.conv8(c7_dense))
        c8_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv8], 1))
        conv9 = self.relu(self.conv9(c8_dense))
        c9_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv9], 1))
        conv10 = self.relu(self.conv10(c9_dense))
        c10_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv10], 1))
        conv11 = self.relu(self.conv11(c10_dense))
        c11_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv10, conv11], 1))
        conv12 = self.relu(self.conv12(c11_dense))
        c12_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv11, conv12], 1))
        return c12_dense

class denseblock_3(nn.Module):
    def __init__(self, in_channels):
        super(denseblock_3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=224, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=224, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(in_channels=288, out_channels=32, kernel_size=3, stride=1, padding=1)        

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv4, conv5], 1))
        conv6 = self.relu(self.conv6(c5_dense))
        c6_dense = self.relu(torch.cat([conv1, conv3, conv5, conv6], 1))
        conv7 = self.relu(self.conv7(c6_dense))
        c7_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv7], 1))
        conv8 = self.relu(self.conv8(c7_dense))
        c8_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv8], 1))
        conv9 = self.relu(self.conv9(c8_dense))
        c9_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv9], 1))
        conv10 = self.relu(self.conv10(c9_dense))
        c10_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv10], 1))
        conv11 = self.relu(self.conv11(c10_dense))
        c11_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv10, conv11], 1))
        conv12 = self.relu(self.conv12(c11_dense))
        c12_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv11, conv12], 1))
        conv13 = self.relu(self.conv13(c12_dense))
        c13_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv10, conv12, conv13], 1))
        conv14 = self.relu(self.conv14(c13_dense))
        c14_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv11, conv13, conv14], 1))
        conv15 = self.relu(self.conv15(c14_dense))
        c15_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv8, conv10, conv12, conv14, conv15], 1))
        conv16 = self.relu(self.conv16(c15_dense))
        c16_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv9, conv11, conv13, conv15, conv16], 1))
        return c16_dense

class denseblock_4(nn.Module):
    def __init__(self, in_channels):
        super(denseblock_4, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv4, conv5], 1))
        conv6 = self.relu(self.conv6(c5_dense))
        c6_dense = self.relu(torch.cat([conv1, conv3, conv5, conv6], 1))
        conv7 = self.relu(self.conv7(c6_dense))
        c7_dense = self.relu(torch.cat([conv1, conv2, conv4, conv6, conv7], 1))
        conv8 = self.relu(self.conv8(c7_dense))
        c8_dense = self.relu(torch.cat([conv1, conv3, conv5, conv7, conv8], 1))
        return c8_dense

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()

        #num_layers = 10, 12, 16, 8
        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()

        self.denseblock1 = self._make_dense_block(denseblock_1, 64)
        self.denseblock2 = self._make_dense_block(denseblock_2, 96)
        self.denseblock3 = self._make_dense_block(denseblock_3, 112)
        self.denseblock4 = self._make_dense_block(denseblock_4, 144)

        self.transitionlayer1 = self._make_transition_layer(transitionlayer, in_channels=192, out_channels=96)
        self.transitionlayer2 = self._make_transition_layer(transitionlayer, in_channels=224, out_channels=112)
        self.transitionlayer3 = self._make_transition_layer(transitionlayer, in_channels=288, out_channels=144)
        self.transitionlayer4 = self._make_transition_layer(transitionlayer, in_channels=160, out_channels=128)

        self.bn = nn.BatchNorm2d(num_features=128)
        self.pre_classifier = nn.Linear(128 * 2 * 2, 512)
        self.classifier = nn.Linear(512, num_classes)

    def _make_dense_block(self, block, in_channels):
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x):
        out = self.relu(self.firstconv(x))
        out = self.denseblock1(out)
        out = self.transitionlayer1(out)
        out = self.denseblock2(out)
        out = self.transitionlayer2(out)
        out = self.denseblock3(out)
        out = self.transitionlayer3(out)
        out = self.denseblock4(out)
        out = self.transitionlayer4(out)        
        out = self.bn(out)
        out = out.view(-1, 128 * 2 * 2)
        out = self.pre_classifier(out)
        out = self.classifier(out)
        return out

def densenet53_sht1():
    return DenseNet(10)
