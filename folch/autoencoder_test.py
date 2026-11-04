import sys
sys.path.insert(0, '/home/ipa/quanz/user_accounts/espalding/dependencies')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from modules_autoencoder import autoenc, FizeauDataset
from astropy.io import fits

training_frac = 0.8
N = 100
n_training = int(N*training_frac)
learning_rates = np.array([0.01, 0.001, 0.0001])
batch_size = 20
training_epochs = 100

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

simulated_images_varied_OPD = '/home/ipa/quanz/user_accounts/espalding/fake_datasets/opd_only/'

data = Path(simulated_images_varied_OPD)

data = sorted(data.glob("*.fits"))

# load the dataset, and scale logarithmically
dataset = FizeauDataset(data, log=True)

idxs = np.arange(0, N)
np.random.shuffle(idxs)

# define which frames to use for training and testing
train_idxs = idxs[:n_training]
test_idxs = idxs[n_training:]

train_dataset = Subset(dataset, train_idxs)
test_dataset = Subset(dataset, test_idxs)

# batch size: number of images to pass through the network at once; these get shuffled if shuffle=True
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# test loader does not call backward or optimizer step; batch size only decides how many images are stacked for 1 forward pass
# batch_size=1 makes each image its own tensor
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(len(dataset))

example_frame = dataset[0][0].numpy().squeeze()

# loss: mean absolute error (OK for 2D image)
loss = nn.L1Loss()

# build untrained autoencoder whose input has one image channel
# channel: number of values at each pixel (RGB would be 3)
model = autoenc(1)

# get first image from training set
lol = train_loader.dataset[0][0]

# pass image through autoencoder
latent, denoised, layer_reps = model(lol)

'''
print(len(layer_reps))
plt.imshow(layer_reps[4][0].detach().numpy())
print(latent.shape, denoised.shape)
'''

'''
fig, ax = plt.subplots(1, 2)
ax[0].imshow(denoised.detach().numpy()[0, :, :])
ax[1].imshow(latent.detach().numpy()[0, :, :])
ax[0].set_title("reconstructed")
ax[1].set_title("latent")
'''

loss_ = loss(lol, denoised)
print(loss_.detach().numpy())


loss_lr = np.zeros((3, 4*training_epochs))
models = []

# for i, learning_rate in enumerate([learning_rates[2]]):
for i, learning_rate in enumerate(learning_rates):
    print(f"learning rate: {learning_rate}")
    model = autoenc(1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)


    loss_per_epoch = []

    for epoch in range(training_epochs):
        print(f"Epoch [{epoch + 1}/{training_epochs}]")
        for batch_index, (data, _) in enumerate(tqdm(iter(train_loader))):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)

            # Forward pass: compute the model output
            latent, reconstructed, _ = model(data)
            loss_ = loss(reconstructed, data)
            loss_per_epoch.append(loss_.detach().numpy())
            
            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss_.backward()

            # Optimization step: update the model parameters
            optimizer.step()

    loss_lr[i, :] = np.array(loss_per_epoch)
    models.append(model)
'''
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_figheight(7)
fig.set_figwidth(25)
epochs = np.linspace(0, 100, num=400)
ax[0].plot(epochs, loss_lr[0, :])
ax[0].set_title("learning rate = 0.01")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("L1 loss")
ax[1].plot(epochs, loss_lr[1, :])
ax[1].set_title("learning rate = 0.001")
ax[2].set_xlabel("epochs")

ax[2].plot(epochs, loss_lr[2, :])
ax[2].set_title("learning rate = 0.0001")
ax[2].set_xlabel("epochs")
'''

out_root = Path("/Users/eckhartspalding/Documents/git.repos/ml/folch/reconstructions")
subset = test_loader.dataset

for learning_rate, model in zip(learning_rates, models):
    model.eval()
    lr_dir = out_root / f"lr_{learning_rate:.0e}"
    lr_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, test_data in enumerate(subset):
            test_img, opd = test_data
            latent, reconstructed, layer_reps = model(test_img)

            src = Path(subset.dataset.files[subset.indices[i]])
            hdr = fits.Header()
            hdr["OPD"] = float(opd)
            hdr["LR"] = float(learning_rate)
            hdr["SOURCE"] = src.name

            fits.writeto(lr_dir / f"{src.stem}_original.fits", test_img.detach().numpy()[0], hdr, overwrite=True)
            fits.writeto(lr_dir / f"{src.stem}_reconstructed.fits", reconstructed.detach().numpy()[0], hdr, overwrite=True)

    print(f"wrote {len(subset)} frame pairs to {lr_dir}")

fig, ax = plt.subplots(len(subset), 3)
fig.set_figheight(60)
fig.set_figwidth(7.5)

model = models[-1]
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(subset):
        test_img = test_data[0]
        latent, reconstructed, layer_reps = model(test_img)

        '''
        ax[i, 0].imshow(reconstructed.detach().numpy()[0, :, :])
        ax[i, 1].imshow(test_img.detach().numpy()[0, :, :])
        ax[i, 2].imshow(latent.detach().numpy()[0, :, :])
        ax[i, 0].set_title("reconstructed")
        ax[i, 1].set_title("original")
        ax[i, 2].set_title("latent representation")
        '''



