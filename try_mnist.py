# %%
import matplotlib.pyplot as plt
from PIL import Image
from nn2 import NeuNet, sigmoid, LayerModel, tanh
from numpy import array

# %%
MAX_NUM = 6
MIN_NUM = 1

nums = {i: Image.open(f"font/{i}.bmp") for i in range(MIN_NUM, MAX_NUM+1)}

# plt.imshow(nums[1], cmap='gray')   # type: ignore
# plt.show()

# %%
h1 = LayerModel((MAX_NUM-MIN_NUM+1)*2)
h1.activation(tanh)
h2 = LayerModel((MAX_NUM-MIN_NUM+1))
h2.activation(sigmoid)
nn = NeuNet(64)
nn.add_layer(h1)
nn.add_layer(h2)

# %%
x_train = array([
    num_img.getdata() for num_img in nums.values()
])/255
y_train = array([
    [0.]*(num-MIN_NUM)+[1.]+[0.]*(MAX_NUM-num+MIN_NUM-1) for num in nums.keys()
])

# %%
nn.fit(x_train, y_train, learning_rate=3, passes=1000, batch_size=4)
