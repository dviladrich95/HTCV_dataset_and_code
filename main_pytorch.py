import numpy as np
import tqdm
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gaussian_logit_sampling import GaussianLogitSampler
from dataloader import DataLoader

LEARNING_RATE = 0.0001
BETA_1 = 0.9

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def res_block(input_tensor, res_func, num_of_output_feature_maps):
    input_tensor = nn.Conv2d(input_tensor.shape[1],num_of_output_feature_maps, 1).to(input_tensor.device)(input_tensor)
    res_func = torch.add(input_tensor, res_func)
    res_func = F.relu(res_func)
    return res_func

class generator(nn.Module):
    def __init__(self, nCPI, nI):
        super(generator, self).__init__()

        # nCPI,nI = 24, 256*128
        self.conv1_1_ = nn.Conv2d(nCPI * nI, 64 * 2, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1)
        self.conv1_5 = nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(64 * 2, 128 * 2, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(128 * 2, 128 * 2, 3, 1, 1)
        self.conv3_5 = nn.Conv2d(128 * 2, 128 * 2, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(128 * 2, 256 * 2, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(256 * 2, 256 * 2, 3, 1, 1)
        self.conv5_5 = nn.Conv2d(256 * 2, 256 * 2, 3, 1, 1)

        self.conv7_1 = nn.Conv2d(256 * 2, 128 * 2, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(128 * 2, 128 * 2, 3, 1, 1)
        self.conv7_5 = nn.Conv2d(128 * 2, 128 * 2, 3, 1, 1)

        self.conv9_1 = nn.Conv2d(128 * 2, 64 * 2, 3, 1, 1)
        self.conv9_3 = nn.Conv2d(64 * 2, 64, 3, 1, 1)
        self.conv9_5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9_7 = nn.Conv2d(64, 24 * 2, 3, 1, 1)

        self.droupout = nn.Dropout2d(p=0.2)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self._initialize_weights()

    def forward(self, x):
        # operations to perform
        x1 = F.relu(self.conv1_1_(x))

        x11 = F.relu(self.conv1_3(x1))
        x12 = self.droupout(x11)
        x13 = F.relu(self.conv1_5(x12))
        x2 = self.droupout(x13)
        x21 = torch.add(x1, x2)
        x22 = F.relu(x21)
        x3 = self.maxpool(x22)

        x = F.relu(self.conv3_1(x3))
        x = self.droupout(x)
        x = F.relu(self.conv3_3(x))
        x = self.droupout(x)
        x = F.relu(self.conv3_5(x))
        x = self.droupout(x)
        x6 = res_block(x3, x,128*2)
        x6 = self.maxpool(x6)

        x = F.relu(self.conv5_1(x6))
        x = self.droupout(x)
        x = F.relu(self.conv5_3(x))
        x = self.droupout(x)
        x = F.relu(self.conv5_5(x))
        x = self.droupout(x)
        x9 = res_block(x6, x, 256*2)
        x9 = self.upsampling(x9)

        x = F.relu(self.conv7_1(x9))
        x = self.droupout(x)
        x = F.relu(self.conv7_3(x))
        x = self.droupout(x)
        x = F.relu(self.conv7_5(x))
        x = self.droupout(x)
        x12 = res_block(x9, x, 128*2)
        x12 = self.upsampling(x12)

        x = F.relu(self.conv9_1(x12))
        x = self.droupout(x)

        x = F.relu(self.conv9_3(x))
        x = self.droupout(x)

        x = F.relu(self.conv9_5(x))

        x = self.conv9_7(x)

        x = GaussianLogitSampler()(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


class discriminator(nn.Module):
    def __init__(self, nCPI, nI):
        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(nCPI * (nI+1), 64 * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1)

        self.maxpool = nn.MaxPool2d((2, 2))
        self.dense1 = nn.Linear(128*8*16,1024)
        self.dense2 = nn.Linear(1024,256)
        self.dense3 = nn.Linear(256,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.softmax(self.dense3(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))




netG = generator(24,4).to(device)
netD = discriminator(24,4).to(device)
optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE)
optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE)
criterionG = torch.nn.BCELoss()
criterionD = torch.nn.BCELoss()

batch_size = 1

dataloader = DataLoader(batch_size)



test_examples = 1
test_samples = 5
( data_val_X_f, data_val_X_o, data_val_Y, data_val_Y_) = dataloader.get_val_data(test_examples,test_samples);#, data_X_o

print('Done.')

l_loss_d = []
l_loss_g = []

epochs = 50
# For each epoch
for _ in range(epochs):
    steps = 4000
    start = time.time()
    for step in tqdm.tqdm(range(1, steps+1)):
        # -----------------------------------------------------------------------
        # Train Discriminator for Synthetic Likelihood
        # -----------------------------------------------------------------------

        data_X_s_batch, data_X_o_batch, data_Y_batch = dataloader.train_data_batch()
        netD.zero_grad()

        data_X_s_batch = np.moveaxis(data_X_s_batch,-1,1)
        data_X_s_batch = torch.tensor(data_X_s_batch, device=device,dtype=torch.float32)
        data_Y_batch = np.moveaxis(data_Y_batch,-1,1)
        data_Y_batch = torch.tensor(data_Y_batch, device=device,dtype=torch.float32)

        generated_y = netG(data_X_s_batch)
        data_Y_batch_wc = torch.cat([data_X_s_batch,data_Y_batch], dim=1)
        generated_y_wc = torch.cat([data_X_s_batch,generated_y], dim=1)

        X = torch.cat([data_Y_batch_wc , generated_y_wc],dim=0) #+ np.random.normal(0,0.05,size=y_batch.shape)
        y = np.zeros([2*batch_size,2]) + np.random.uniform(0,0.05,size=(2*batch_size,2))
        y[0:batch_size,1] = 1 - np.random.uniform(0,0.05,size=(batch_size,))
        y[batch_size:,0] = 1 - np.random.uniform(0,0.05,size=(batch_size,))
        d_out = netD(X).view(-1)
        d_err = criterionD(d_out, y)

        l_loss_d.append(d_err.item())
        if len(l_loss_d) > 1000:
            l_loss_d.pop(0)

        d_err.backward()
        optimizerD.step()

        # -----------------------------------------------------------------------
        # Train Generator
        # -----------------------------------------------------------------------
        netG.zero_grad()

        data_X_s_batch, data_X_o_batch, data_Y_batch = dataloader.train_data_batch()

        y2 = np.zeros([batch_size,2])
        y2[:,1] = 1 - np.random.uniform(0,0.05,size=(batch_size,))

        g_out = netG([data_X_s_batch, data_X_o_batch, data_X_s_batch]).view(-1)
        g_err = criterionG(g_out, [data_Y_batch, y2])
        l_loss_g.append(g_err.item())
        if len(l_loss_g) > 1000:
            l_loss_g.pop(0)

        g_err.backward()
        optimizerG.step()


        if step % 10 == 0:
            tqdm.write('Step: ' + str(step) + ' -- Loss - D: ' + str(np.mean(np.array(l_loss_d))) + ' -- Loss - G: ' + str(np.mean(np.array(l_loss_g), axis=0)))
