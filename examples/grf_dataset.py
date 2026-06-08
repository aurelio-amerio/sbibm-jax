#%%
import os 
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax 
from jax import numpy as jnp

from sbibm_jax.data import TaskDataset

import matplotlib.pyplot as plt
#%%
dataset = TaskDataset("gaussian_random_field_256", normalize=True, dtype=jnp.bfloat16, use_prefetching=False)
# %%
train_dataset = dataset.get_train_loader(16)
# %%
data = next(iter(train_dataset))
# %%
data[0].shape, data[1].shape
# %%
plt.imshow(data[1][12,:,:,0], cmap="coolwarm")
plt.show()
# %%
