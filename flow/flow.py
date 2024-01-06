# %%
from utils import *


# %%
# Transformation affine : f(y; s,t) = y * exp(s) + t
class AffineLayer(FlowModule):
    def __init__(self, dim: int):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(dim))
        self.t = nn.Parameter(torch.zeros(dim))

    def encoder(self, y):
        return y * torch.exp(self.s) + self.t, self.s.sum(dim=-1)

    def decoder(self, x):
        return (x - self.t) * torch.exp(-self.s), -self.s.sum(dim=-1)
        # log |det J_f(x)| = -sum(s)


# %%
normal = torch.distributions.Normal(0, 1)
diagn = torch.distributions.Independent(normal, 0)

affine = AffineLayer(1)
model = FlowModel(diagn, affine)

# %%
z = torch.distributions.Normal(0, 1).sample((10, 1))
z
# %%
model.decoder(z), model.decoder(z)[0][0].shape
# %%
model.encoder(torch.randn(10, 1)), model.encoder(torch.randn(10, 1))[1][0].shape

# %%
test = torch.randn(10, 1)
model.decoder(model.encoder(test)[1][0])
# %%
test
# %%
model.plot(test, 10)
# %%
affine.check(test)
# %%
model.plot(test, 5)
# %%
