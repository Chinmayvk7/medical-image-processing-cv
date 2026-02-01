import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
folder_paths = [
    r"C:\Users\chinm\Downloads\DSP Mini_Projects\Unet_x_ray_tumor_detection\malignant",
    r"C:\Users\chinm\Downloads\DSP Mini_Projects\Unet_x_ray_tumor_detection\benign"
]

def conv2d_scratch(x, kernel, bias=None, stride=1, padding='same'):
    x = np.asarray(x)
    added_batch = False
    if x.ndim == 3:
        x = x[np.newaxis, ...]
        added_batch = True
    N, H, W, C = x.shape
    kh, kw, kc, oc = kernel.shape
    assert kc == C, "Kernel in_channels must match input channels"

    if padding == 'same':
        out_h = int(np.ceil(H / stride))
        out_w = int(np.ceil(W / stride))
        pad_h = max((out_h - 1) * stride + kh - H, 0)
        pad_w = max((out_w - 1) * stride + kw - W, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

    x_p = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0.0)
    out = np.zeros((N, out_h, out_w, oc), dtype=np.float32)

    ker_flat = kernel.reshape(-1, oc)
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kh
                w_start = j * stride
                w_end = w_start + kw
                patch = x_p[n, h_start:h_end, w_start:w_end, :]
                patch_flat = patch.reshape(-1)
                out[n, i, j, :] = patch_flat.dot(ker_flat)
                if bias is not None:
                    out[n, i, j, :] += bias
    if added_batch:
        return out[0]
    return out

def maxpool2d_scratch(x, pool_size=2, stride=2, padding='valid'):
    x = np.asarray(x)
    added_batch = False
    if x.ndim == 3:
        x = x[np.newaxis, ...]
        added_batch = True
    N, H, W, C = x.shape
    if isinstance(pool_size, int):
        ph = pw = pool_size
    else:
        ph, pw = pool_size

    if padding == 'same':
        out_h = int(np.ceil(H / stride))
        out_w = int(np.ceil(W / stride))
        pad_h = max((out_h - 1) * stride + ph - H, 0)
        pad_w = max((out_w - 1) * stride + pw - W, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0
        out_h = (H - ph) // stride + 1
        out_w = (W - pw) // stride + 1

    x_p = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=-np.inf)
    out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + ph
                w_start = j * stride
                w_end = w_start + pw
                patch = x_p[n, h_start:h_end, w_start:w_end, :]
                out[n, i, j, :] = patch.max(axis=(0, 1))
    if added_batch:
        return out[0]
    return out






size = 128 # Input size: 128x128

images = [] # Empty list to store the original images in
masks = [] # Empty list to store the masks in

found_mask = False # This flag helps us handle multiple masks for the same image
# Loop through both folders
for folder_path in folder_paths:
    # Loop through all files in the current folder (sorted for consistency)
    for file_path in sorted(glob(folder_path + "/*")):
        # Load and resize the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (size, size)) # Resize image to 128×128
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert RGB to grayscale
        img = img / 255.0 # Normalize to [0,1]

        if "mask" in file_path: # Checks if the filename contains "mask"
            if found_mask:
                # Combine with the previous mask
                masks[-1] += img
                # Ensure binary output (0 or 1)
                masks[-1] = np.where(masks[-1] > 0.5, 1.0, 0.0)
            else:
                masks.append(img) # Adds the first mask to the list
                found_mask = True
        else:
            images.append(img) # Adds original image to the list
            found_mask = False

# Convert lists to NumPy arrays
X = np.array(images) # Create an array of all original images
y = np.array(masks) # Create an array of all masked imaged (Ground truth)

X = np.expand_dims(X, -1)
y = np.expand_dims(y, -1)

print(f"X shape: {X.shape} | y shape: {y.shape}")
# Split images into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Build the U-Net
input_layer = Input(shape=(size, size, 1))
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(input_layer)
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

# Second encoder block
conv2 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(pool1)
conv2 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

# Third encoder block
conv3 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(pool2)
conv3 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)

# Fourth encoder block
conv4 = Conv2D(512, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(pool3)
conv4 = Conv2D(512, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)

# --- Bottleneck ---
bottleneck = Conv2D(1024, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(pool4)
bottleneck = Conv2D(1024, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(bottleneck)

# First decoder block
upconv1 = Conv2DTranspose(512, (2, 2), strides=2, padding="same",kernel_initializer="he_normal")(bottleneck)
concat1 = concatenate([upconv1, conv4])
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(concat1)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv5)

# Second decoder block
upconv2 = Conv2DTranspose(256, (2, 2), strides=2, padding="same",kernel_initializer="he_normal")(conv5)
concat2 = concatenate([upconv2, conv3])
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(concat2)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv6)

# Third decoder block
upconv3 = Conv2DTranspose(128, (2, 2), strides=2, padding="same",kernel_initializer="he_normal")(conv6)
concat3 = concatenate([upconv3, conv2])
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(concat3)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv7)

# Fourth decoder block
upconv4 = Conv2DTranspose(64, (2, 2), strides=2, padding="same",kernel_initializer="he_normal")(conv7)
concat4 = concatenate([upconv4, conv1])
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(concat4)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer="he_normal")(conv8)

# --- Output layer ---
output_layer = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv8)

# --- Model creation ---
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs = 40, validation_data = (X_val,y_val),verbose=1)

# Compute IoU
from sklearn.metrics import jaccard_score
# Compute predicted mask
pred=model.predict(X_val,verbose=1)
pred = (pred > 0.5).astype(int) # binarize
y_true = y_val.astype(int)
# Compute IoU based on flatten predictions and ground truths
iou = jaccard_score(pred.flatten(), y_true.flatten())
print(f" IoU (Jaccard Score): {iou:.4f}")


# Plot
i=6 # Try other values
plt.subplot(1, 3, 1)
plt.imshow(X_val[i],cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(y_val[i],cmap="gray")
plt.subplot(1, 3, 3)
pred=model.predict(np.expand_dims(X_val[i], axis=0),verbose=1)[0]
pred = (pred > 0.5) # binarize
plt.imshow(pred,cmap="gray")

