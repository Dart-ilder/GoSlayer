from tqdm.auto import tqdm
import sys
import os
sys.path.insert(0, "./data/maps_1key_noaug/processed/Depth-Anything-V2")
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import Resize, GaussianBlur
from depth_anything_v2.dpt import DepthAnythingV2




class ImageMapsDataset(Dataset):
    def __init__(self, images_path, points_path):
        super().__init__()
        self.images_path = images_path
        self.points_path = points_path
        
    def __len__(self):
        return (len(os.listdir(self.images_path)) - 3) * 10
    
    def _prepare_map(self, points):
        """
        Makes Top-bottom view of pointcloud
        """
        x = points[:, 0]
        y = points[:, 2]

        x = ((x - x.min())/(x.max() - x.min())) * 319 / 2 + 80
        x = x.astype(np.int16)

        y = ((y - y.min())/(y.max() - y.min())) * 239 / 2 + 90
        y = y.astype(np.int16)

        pic = np.ones((240, 320))
        pic[-y, x] = np.zeros(len(y))
        
        return pic
    

    def __getitem__(self, index):
        i = index // 10
        j = index % 10
        
        data = np.load(f'{self.images_path}/map{i+1}_data.npz')
        
        points = np.asarray(o3d.io.read_point_cloud(f"{self.points_path}/map{i+1}_{j}.ply").points)
        
        im = data['images'][0,j,0]
        map = data['maps'][0,j,0]

        im = torch.from_numpy(im) / 255
        im = im.permute(0, 3, 1, 2)

        map = torch.from_numpy(map[:, :, 0]).unsqueeze(0) / 255
        points_map = torch.from_numpy(self._prepare_map(points)).unsqueeze(0).to(torch.float32)
        
        return im, points_map, map
    
    
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()    
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(features, 800),
            nn.ReLU(),
            nn.Linear(800, 400)
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    
    
class ImagesEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()    
        self.encoder1 = EncoderBlock(in_channels, out_channels, features)
        self.encoder2 = EncoderBlock(in_channels, out_channels, features)
        self.encoder3 = EncoderBlock(in_channels, out_channels, features)
    def forward(self, x1, x2, x3):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder3(x3)
        return x1, x2, x3




class MapGenerator(nn.Module):
    def __init__(self, depth_model=None):
        super().__init__()
        
        self.depth_model = depth_model
        self.resize = Resize((210, 280))
        
        self.im_encoder = ImagesEncoder(3, 16, 4_800)
        self.depth_encoder = ImagesEncoder(1, 16, 3_536)
        self.points_encoder = EncoderBlock(1, 16, 4_800)
        
        self.make_embedding = nn.Sequential(
            nn.Linear(2800, 3_200),
            nn.ReLU(),
            nn.Linear(3_200, 4_800)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=2, stride=2),
        )

    def forward(self, ims, points):
        im1, im2, im3 = ims[:, 0, :], ims[:, 1, :], ims[:, 2, :]
        
        with torch.no_grad():
            d1, d2, d3 = self.depth_model(self.resize(im1)), self.depth_model(self.resize(im2)), self.depth_model(self.resize(im3))
        
        im1, im2, im3 = self.im_encoder(im1, im2, im3)
        d1, d2, d3 = self.depth_encoder(d1.unsqueeze(1), d2.unsqueeze(1), d3.unsqueeze(1))
        
        points = self.points_encoder(points)
        x = torch.cat([im1, im2, im3, d1, d2, d3, points], axis=1)
        
        x = self.make_embedding(x)
        x = x.view(-1, 16, 15, 20)
        x = self.decoder(x)
        return x
    
     
    
def gauss_pixel_loss(y_true, y_pred, loss_fn, gb):
    EPS = 1e-7
    y_true = gb(y_true)
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min() + EPS)
    y_true[y_true < 0.9] = 0
    return loss_fn(y_pred, y_true)

    
def train(model, max_epoch, train_loader, optimizer, loss_fn, gb, device='cpu'):
    losses = []
    for _ in tqdm(range(max_epoch)):
        model.train()
        for ims, points, maps in train_loader:
            optimizer.zero_grad()
            ims, points, maps = ims.to(device), points.to(device), maps.to(device)
            output = model(ims,points)
            output = torch.sigmoid(output)
            loss = gauss_pixel_loss(maps, output, loss_fn, gb)
            print("loss: ", loss.detach().cpu().numpy())
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
            
    return losses


def main():
    
    message = "Write path to save model's weights.\nIf you don't need save, write 'n'\n"
    savepath = input(message)
    
    
    images_path = "./data/maps_1key_noaug/processed"
    points_path = "./data/point_clouds/point_clouds11"
    train_dataset = ImageMapsDataset(images_path, points_path)
    
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load('./data/maps_1key_noaug/processed/Depth-Anything-V2/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_model.eval()
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.BCELoss()
    res = Resize((210, 280))
    gb = GaussianBlur(kernel_size=15, sigma=5)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=250)
    model = MapGenerator(depth_model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    
    max_epoch = 10
    
    losses = train(model, max_epoch, train_loader, optimizer, loss_fn, gb, device=device)
    
    if savepath != 'n':
        torch.save(model.state_dict(), savepath)
        
        
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()