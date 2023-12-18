# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datamaestro import prepare_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()


class MnistDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = torch.FloatTensor(data)
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


train_dataset = MnistDataset(train_images, train_labels)
test_dataset = MnistDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# %%
x, y = next(iter(train_loader))
plt.imshow(x[0], cmap="gray")


# %%
class Encoder(nn.Module):
    def __init__(self, in_size, hid_size, latent_dim) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_size, hid_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_size, latent_dim)
        self.fc3 = nn.Linear(hid_size, latent_dim)

    def forward(self, x):
        """Encodes the image x into the latent dimension

        Args:
            x (Tensor): Image
        """
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.relu(x)
        mu = self.fc2(x)
        sigma = self.fc3(x)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, hid_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, z):
        """Takes a sample in the latent dimension and decodes it to output
        an image

        Args:
            z (Tensor): Latent vector
        """
        x = self.relu(self.fc(z))
        x = self.fc2(x)
        return x


# %%
enc = Encoder(784, 100, 50)
enc(x[0].unsqueeze(0))


# %%
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl = 0

    def forward(self, x):
        """Forward pass of the VAE,
        - Encodes the image in the latent space
        - Samples z from the latent space
        - Decodes the image

        Args:
            x (Tensor): Image
        """
        mu, sigma = self.encoder(x)

        z = mu + torch.distributions.Normal(0, 1).sample(mu.shape) * sigma
        # x_hat = self.decoder(z.squeeze(0))

        self.kl = (1 / 2) * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        return x_hat, mu, sigma


# %%
enc = Encoder(784, 100, 50)
dec = Decoder(50, 100, 784)

vae = VAE(enc, dec)

# %%
plt.imshow(vae(x[0].unsqueeze(0))[0].detach().numpy().reshape((28, 28)))
# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

for epoch in tqdm(range(50)):
    l = 0
    for i, (x, _) in enumerate(train_loader):
        x_hat, mu, sigma = vae(x)
        x_hat = x_hat.view(x.size(0), x.size(1), x.size(2))
        loss = criterion(x, x_hat) - vae.kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss.item()

    print(f"Epoch {epoch}, loss {l/i}")

# %%
x, y = next(iter(train_loader))

plt.imshow(x[31])
# %%
plt.imshow(vae(x[31].unsqueeze(0))[0].detach().numpy().reshape((28, 28)))

# %%
