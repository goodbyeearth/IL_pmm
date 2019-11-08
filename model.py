import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv256_v1(nn.Module):
    def __init__(self):
        from featurize import get_feature_shape_v1
        super(Conv256_v1, self).__init__()
        self.conv1 = nn.Conv2d(get_feature_shape_v1()[0], 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(11*11*256, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        output = self.fc(x)

        return output


class Conv256_v2(nn.Module):
    def __init__(self):
        from featurize import get_feature_shape_v1
        super(Conv256_v2, self).__init__()
        self.conv1 = nn.Conv2d(get_feature_shape_v1()[0], 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(11*11*256, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)

        output = self.fc(x)

        return output

# cnn 3
class FCConv(nn.Module):
    def __init__(self):
        from featurize import get_feature_shape_v2
        super(FCConv, self).__init__()
        self.conv1 = nn.Conv2d(get_feature_shape_v2()[0], 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(11 * 11 * 256 + 3, 6)

    def forward(self, x):
        c = F.relu(self.bn1(self.conv1(x['board'])))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))
        c = c.view(c.size(0), -1)

        s = x['flat']
        # print(c.shape)
        # print(s.shape)
        vec_cat = torch.cat([c, s], dim=1)

        output = self.fc(vec_cat)

        return output


class FCConv_v2(nn.Module):
    def __init__(self):
        from featurize import get_feature_shape_v3
        super(FCConv_v2, self).__init__()
        self.conv1 = nn.Conv2d(get_feature_shape_v3()[0], 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(11 * 11 * 256 + 14, 6)

    def forward(self, x):
        c = F.relu(self.bn1(self.conv1(x['board'])))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))
        c = c.view(c.size(0), -1)

        s = x['flat']

        vec_cat = torch.cat([c, s], dim=1)

        output = self.fc(vec_cat)

        return output


# cnn 5
class FCConv5(nn.Module):
    def __init__(self):
        from featurize import get_feature_shape_v2
        super(FCConv5, self).__init__()
        self.conv1 = nn.Conv2d(get_feature_shape_v2()[0], 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(11 * 11 * 256 + 3, 6)

    def forward(self, x):
        c = F.relu(self.bn1(self.conv1(x['board'])))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))
        c = F.relu(self.bn4(self.conv4(c)))
        c = F.relu(self.bn5(self.conv5(c)))
        c = c.view(c.size(0), -1)

        s = x['flat']
        # print(c.shape)
        # print(s.shape)
        vec_cat = torch.cat([c, s], dim=1)

        output = self.fc(vec_cat)

        return output
