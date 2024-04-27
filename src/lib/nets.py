import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .arg_parse import parse_arguments
import segmentation_models_pytorch as smp # https://pypi.org/project/segmentation-models-pytorch/0.0.3/
from tqdm import tqdm


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))

        self.bridge = Bridge(512)

        self.decoderBlocks = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 1)
        ])

    def forward(self, x):
        x = self.resnet(x)
        x = self.bridge(x)

        for block in self.decoderBlocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, channels_into_upconv, channels_out):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(channels_into_upconv, channels_out, kernel_size=2, stride=2)
        self.conv = ConvBlock(channels_out)

    def forward(self, inputs):
        x = self.up_conv(inputs)
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels)
        )

    def forward(self, x):
        return self.bridge(x)


class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()
        args = parse_arguments()
        self.dropout = nn.Dropout(0.5)
        self.dropout_toggle = args.dropout
        self.input_channels_num = 1 if args.dataset == 0 else 3
        self.output_num = 100 if args.dataset == 2 else 10

    def get_flat_params(self, device):
        params = [param.data for _, param in self.named_parameters()]
        flat_params = torch.cat([torch.flatten(param) for param in params]).to(device)
        return flat_params

    def load_from_flat_params(self, f_params):
        shapes = [(name, param.shape, param.numel()) for name, param in self.named_parameters()]
        state = {}
        c = 0
        for name, tsize, tnum in shapes:
            state[name] = torch.nn.Parameter(f_params[c: c + tnum].reshape(tsize))
            c += tnum
        self.load_state_dict(state, strict=True)
        return self

"""
class ModifiedLeNet(BaseNN):
    def __init__(self):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(self.input_channels_num, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 10, 3, 1)
        self.conv3 = nn.Conv2d(10, 10, 3, 1)
        self.fc1 = nn.Linear(10 * 2 * 2, self.output_num)

    def forward(self, x, test_dropout=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        if self.dropout_toggle == 1 and not test_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(self.fc1(x), dim=1)
"""
"""
def train(model, train_loader, optimizer, device, train_loss_file):
    model.train()
    train_loss = 0

    for (data, target) in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()# * data.size(0)
        loss.backward()
        optimizer.step()

        train_loss_file.write(f"{str(float(loss.data.cpu().numpy()))}\n")

    return train_loss / len(train_loader.dataset)
"""

def train(model, train_loader, optimizer, device, train_loss_file):
    model.train()
    train_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for _, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        loss = F.mse_loss(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        # Update progress bar description with current loss
        progress_bar.set_description(f'Train Loss: {loss.item()}')

        train_loss_file.write(f"{str(float(loss.data.cpu().numpy()))}\n")

    return train_loss / len(train_loader.dataset)
"""
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += (output.argmax(dim=1) == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy
"""

def test(model, test_loader, device):
    model.eval()
    test_loss = 0

    # Initialize tqdm to track progress
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.mse_loss(output, target)
            test_loss += loss.item()
            
            progress_bar.set_description(f'Test Loss: {loss.item()}')

    test_loss /= len(test_loader.dataset)

    return test_loss