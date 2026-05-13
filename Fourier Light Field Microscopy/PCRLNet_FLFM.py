import tensorflow as tf
import scipy.io as sio
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
# %% # --- Metric functions definition ---
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis = -1)))) / 2.303

# %% # --- Custom Sparse Layers ---
class SparseVoxelPixelLayer(Layer):
    def __init__(self, voxel_indices, pixel_indices, num_voxels, num_pixels, initial_weights, learnable_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.voxel_indices = tf.convert_to_tensor(voxel_indices, dtype=tf.int64)
        self.pixel_indices = tf.convert_to_tensor(pixel_indices, dtype=tf.int64)
        self.num_voxels = num_voxels
        self.num_pixels = num_pixels
        self.num_connections = self.voxel_indices.shape[0]
        self.learnable_weights = learnable_weights
        self.initial_weights = initial_weights
        

    def build(self, input_shape):
        if self.learnable_weights:
            self.edge_weights = self.add_weight(
                shape=(self.num_connections,),
                initializer=tf.constant_initializer(self.initial_weights),
                trainable=True,
                name="edge_weights",
                dtype=tf.float32,
                # regularizer=tf.keras.regularizers.l2(1e-2),
                # constraint=MaxNorm(max_value=1.0)
            )
        else:
            self.edge_weights = tf.ones((self.num_connections,), dtype=tf.float32)

        self.sparse_indices = tf.stack([self.pixel_indices, self.voxel_indices], axis=1)
        self.sparse_shape = [self.num_pixels, self.num_voxels]


    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.num_voxels])
        # edge_weights = tf.nn.softplus(self.edge_weights)
        edge_weights = tf.clip_by_value(self.edge_weights, -1.0, 1.0)
        sparse_mat = tf.sparse.SparseTensor(
            indices=self.sparse_indices,
            values=edge_weights,
            dense_shape=self.sparse_shape
        )
        sparse_mat = tf.sparse.reorder(sparse_mat)
        output = tf.sparse.sparse_dense_matmul(sparse_mat, tf.transpose(inputs))
        return tf.transpose(output)

class SparsePixelVoxelLayer(Layer):
    def __init__(self, voxel_indices, pixel_indices, num_voxels, num_pixels, initial_weights, learnable_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.voxel_indices = tf.convert_to_tensor(voxel_indices, dtype=tf.int64)
        self.pixel_indices = tf.convert_to_tensor(pixel_indices, dtype=tf.int64)
        self.num_voxels = num_voxels
        self.num_pixels = num_pixels
        self.num_connections = self.voxel_indices.shape[0]
        self.learnable_weights = learnable_weights
        self.initial_weights = initial_weights

    def build(self, input_shape):
        
        if self.learnable_weights:
            self.edge_weights = self.add_weight(
                shape=(self.num_connections,),
                initializer=tf.constant_initializer(self.initial_weights),
                trainable=True,
                name="edge_weights_back",
                dtype=tf.float32,
                # regularizer=tf.keras.regularizers.l2(1e-2),
                # constraint=MaxNorm(max_value=1.0)
            )
        else:
            self.edge_weights = tf.ones((self.num_connections,), dtype=tf.float32)

        self.sparse_indices = tf.stack([self.voxel_indices, self.pixel_indices], axis=1)
        self.sparse_shape = [self.num_voxels, self.num_pixels]

    def call(self, inputs):
        # tf.debugging.assert_all_finite(self.edge_weights, "Edge weights contain NaN or Inf")
        # tf.print("Input to SparsePixelVoxelLayer NaN check:", tf.reduce_any(tf.math.is_nan(inputs)))

        inputs = tf.reshape(inputs, [-1, self.num_pixels])
        # edge_weights = tf.nn.softplus(self.edge_weights)
        edge_weights = tf.clip_by_value(self.edge_weights, -1.0, 1.0)
        sparse_mat = tf.sparse.SparseTensor(
            indices=self.sparse_indices,
            values=edge_weights,
            dense_shape=self.sparse_shape
        )
        sparse_mat = tf.sparse.reorder(sparse_mat)

        output = tf.sparse.sparse_dense_matmul(sparse_mat, tf.transpose(inputs))
        # tf.debugging.check_numerics(output, message="Backward output has NaNs")

        return tf.transpose(output)
# %% # --- Layers for Image Regularization ---
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

class MedianFilter2D(tf.keras.layers.Layer):
    def __init__(self, filter_size=3):
        super().__init__()
        self.filter_size = filter_size

    def call(self, x):
        """
        x: [B, H, W, C]
        returns: median-filtered tensor [B, H, W, C]
        """
        return tfa.image.median_filter2d(x, filter_shape=(self.filter_size, self.filter_size))

class MedianFilter3D(tf.keras.layers.Layer):
    def __init__(self, filter_size=3):
        super().__init__()
        self.filter_size = filter_size

    def call(self, x):
        """
        Apply 3D median filter to x.
        x: Tensor of shape [B, D, H, W, C]
        returns: median-filtered tensor of same shape
        """
        k = self.filter_size
        # Extract patches: shape → [B, D, H, W, k^3 * C]
        patches = tf.image.extract_volume_patches(
            input=x,
            ksizes=[1, k, k, k, 1],
            strides=[1, 1, 1, 1, 1],
            padding='SAME'
        )

        # Reshape to [B, D, H, W, k^3, C]
        C = tf.shape(x)[-1]
        patches = tf.reshape(patches, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], k * k * k, C])

        # Compute median along patch axis
        median = tfp.stats.percentile(patches, 50.0, axis=4)

        return median

class MedianFilter3DDepthwise(tf.keras.layers.Layer):
    def __init__(self, filter_size=5):
        super().__init__()
        self.filter_size = filter_size

    def call(self, x):
        """
        x: Tensor of shape [B, D, H, W, C]
        returns: median-filtered tensor of same shape
        """
        # Split along depth axis
        depth_slices = tf.unstack(x, axis=1)  # List of [B, H, W, C] tensors

        # Apply 2D median filter to each slice
        filtered_slices = [
            tfa.image.median_filter2d(slice, filter_shape=(self.filter_size, self.filter_size))
            for slice in depth_slices
        ]

        # Stack back along depth axis
        return tf.stack(filtered_slices, axis=1)

class GaussianBlur2DDepthwise(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        # Create a 2D Gaussian kernel
        k = self.kernel_size
        sigma = self.sigma
        radius = k // 2
        x = tf.range(-radius, radius + 1, dtype=tf.float32)
        x, y = tf.meshgrid(x, x)
        g = tf.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        g = g / tf.reduce_sum(g)
        g = g[:, :, tf.newaxis, tf.newaxis]  # shape [k, k, 1, 1]
        self.kernel = self.add_weight(
        name='gaussian_kernel',
        shape=g.shape,
        initializer=tf.constant_initializer(g.numpy()),
        trainable=False
        )

    def call(self, x):
        """
        x: [B, D, H, W, C] — 3D volume
        Applies 2D Gaussian blur to each depth slice independently
        """
        B, D, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]

        # Reshape to [B*D*C, H, W, 1] to apply conv2d per-slice
        x_reshaped = tf.transpose(x, [0, 4, 1, 2, 3])  # [B, C, D, H, W]
        x_reshaped = tf.reshape(x_reshaped, [B * C * D, H, W, 1])

        # Apply conv2d
        blurred = tf.nn.conv2d(x_reshaped, self.kernel, strides=1, padding='SAME')

        # Reshape back to [B, D, H, W, C]
        blurred = tf.reshape(blurred, [B, C, D, H, W])
        blurred = tf.transpose(blurred, [0, 2, 3, 4, 1])  # back to [B, D, H, W, C]

        return blurred

# % Keras Conv3D layer identity initialization
import numpy as np
from tensorflow.keras import initializers

def identity_conv3d_init(shape, dtype=None):
    # shape = (kernel_size_x, kernel_size_y, kernel_size_z, in_ch, out_ch)
    kernel = np.zeros(shape, dtype=np.float32)
    center = (shape[0]//2, shape[1]//2, shape[2]//2)
    for c in range(min(shape[3], shape[4])):  # in_ch, out_ch
        kernel[center[0], center[1], center[2], c, c] = 1.0
    return kernel
def identity_conv2d_init(shape, dtype=None):
    # shape = (kernel_size_x, kernel_size_y, in_ch, out_ch)
    kernel = np.zeros(shape, dtype=np.float32)
    center = (shape[0] // 2, shape[1] // 2)
    for c in range(min(shape[2], shape[3])):  # in_ch, out_ch
        kernel[center[0], center[1], c, c] = 1.0
    return kernel
# conv = tf.keras.layers.Conv3D(1, 3, padding='same', kernel_initializer=identity_conv3d_init, use_bias=False)
# %--- Regularizer Network ---
class RegularizerNetwork_B(Model):
    def __init__(self, voxel_shape):
        super().__init__()
        self.reshape_layer = tf.keras.layers.Reshape(voxel_shape + (1,))
        self.conv1 = tf.keras.layers.Conv3D(4, 11, padding='same', kernel_initializer=identity_conv3d_init, activation='tanh')
        self.conv2 = tf.keras.layers.Conv3D(4, 7, padding='same', kernel_initializer=identity_conv3d_init, activation='tanh')
        self.conv3 = tf.keras.layers.Conv3D(1, 3, kernel_initializer=identity_conv3d_init, padding='same', activation='tanh')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.MedianFilter3DDepthwise = MedianFilter3DDepthwise(filter_size=5)
        self.GaussianBlur2DDepthwise = GaussianBlur2DDepthwise(kernel_size=3, sigma=0.5)
        
    def call(self, x):
        x = self.reshape_layer(x)
        # x = self.MedianFilter3DDepthwise(x)
        # x = self.GaussianBlur2DDepthwise(x)
        # x = MedianFilter3D()(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten_layer(x)
        return x
# %--- Regularizer Network ---
class RegularizerNetwork_F(Model):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape  # (height, width)
        self.reshape_layer = tf.keras.layers.Reshape(image_shape + (1,))
        self.conv1 = tf.keras.layers.Conv2D(4, 11, padding='same', kernel_initializer=identity_conv2d_init, activation='tanh')
        self.conv2 = tf.keras.layers.Conv2D(4, 7, padding='same', kernel_initializer=identity_conv2d_init, activation='tanh')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, kernel_initializer=identity_conv2d_init, padding='same', activation='tanh')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.MedianFilter2D = MedianFilter2D()
    def call(self, x):
        x = self.reshape_layer(x)
        # x = self.MedianFilter2D(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten_layer(x)
        return x
# %
import tensorflow_probability as tfp
# %--- Unrolled RL Model with Regularizer ---
class UnrolledRLWithRegularizer(Model):
    def __init__(self, forward_layer, backward_layer, voxel_shape, image_shape, num_iter=5):
        super().__init__()
        self.forward = forward_layer
        self.backward = backward_layer
        self.num_iter = num_iter
        self.voxel_shape = voxel_shape
        self.image_shape = image_shape
        # self.regularizer = RegularizerNetwork(voxel_shape)
        self.regularizer_B = RegularizerNetwork_B(voxel_shape)
        self.regularizer_F = RegularizerNetwork_F(image_shape)


    def call(self, inputs):
        B = tf.random.uniform(
            shape=(tf.shape(inputs)[0], self.backward.num_voxels, 1),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32
        )
        # B = tf.ones((tf.shape(inputs)[0], self.backward.num_voxels, 1), dtype=tf.float32)
        for _ in range(self.num_iter):
            F_est = self.forward(B)
            F_est = tf.where(tf.math.is_nan(F_est), tf.zeros_like(F_est), F_est)
            # print(tf.reduce_max(F_est))
            F_est=F_est - self.regularizer_F(F_est)
            F_percentile_val = tfp.stats.percentile(F_est, 99.5, axis=[1], keepdims=True)
            F_est = F_est - tf.math.reduce_min(F_est, axis=1)
            F_est_max = tf.reduce_max(F_est, axis=1, keepdims=True)  # shape: (batch_size, 1)
            F_est = tf.math.divide_no_nan(F_est, F_est_max)
            F_est = tf.where(tf.math.is_nan(F_est), tf.zeros_like(F_est), F_est)
            ratio = tf.math.divide_no_nan(inputs, F_est+ 1e-6)
            # tf.print("Before backward, input NaN:", tf.reduce_any(tf.math.is_nan(ratio)))
            correction = self.backward(ratio)
            correction = tf.expand_dims(correction, axis=-1)
            correction = tf.where(tf.math.is_nan(correction), tf.ones_like(correction), correction)
            correction_percentile_val = tfp.stats.percentile(correction, 99.5, axis=[1], keepdims=True)
            correction = tf.clip_by_value(correction, -1e2, correction_percentile_val)
            B = B * correction
            B = tf.where(tf.math.is_nan(B), tf.zeros_like(B), B)
            # B = tf.nn.relu(B)

            # Apply learned regularization
            B = B - tf.expand_dims(self.regularizer_B(tf.squeeze(B, axis=-1)), axis=-1)
            # B_percentile_val = np.percentile(B, 99)
            B_percentile_val = tfp.stats.percentile(B, 99.5, axis=[1], keepdims=True)
            # B = tf.clip_by_value(B,0,B_percentile_val)
            B = B - tf.math.reduce_min(B, axis=1)
            B_max = tf.reduce_max(B, axis=1, keepdims=True)  # shape: (batch_size, 1)
            B = tf.where(tf.math.is_nan(B), tf.zeros_like(B), B)
            B = tf.math.divide_no_nan(B, B_max)
            B = tf.where(tf.math.is_nan(B), tf.zeros_like(B), B)
            # print(F_est.shape)
        return [F_est, tf.squeeze(B, axis=-1)]

# %% # %--- Load .mat file --- 
import mat73
mat = sio.loadmat('data_FLFM.mat')
images = mat['X'].astype('float32')
objects = mat['Y'].astype('float32')
images=images/255
objects=objects/255

images = np.expand_dims(images, 3)

images=images/np.max(images)
objects=objects/np.max(objects)

# Dimensions
ox, oy, oz = objects.shape[1:4]  # object dimensions
ix, iy = images.shape[1:3]       # image dimensions
num_voxels = ox * oy * oz
num_pixels = ix * iy
# %
mat = sio.loadmat('data_FLFM_index_th08.mat')
voxel_coords = mat['voxel_coords'].astype('int64').flatten()
pixel_coords = mat['pixel_coords'].astype('int64').flatten()
F_weights_init = mat['F_weights_init'].astype('float32').flatten()
B_weights_init = mat['F_weights_init'].astype('float32').flatten()

# % Train data
X_train = images.reshape((-1, num_pixels)).astype(np.float32)
Y_train = objects.reshape((-1, num_voxels)).astype(np.float32)
# %
# doff = 1
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='output_2_loss',patience=3,verbose=1,restore_best_weights=True,min_delta=1e-4)

# %
def ssim_F(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, images.shape[1], images.shape[2], 1])  # batch, H, W, 1
    y_pred = tf.reshape(y_pred, [-1, images.shape[1], images.shape[2], 1])
    y_pred=tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    y_true=tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
def ssim_B(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, objects.shape[1], objects.shape[2], objects.shape[3]])  # batch, H, W, 1
    y_pred = tf.reshape(y_pred, [-1, objects.shape[1], objects.shape[2], objects.shape[3]])
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# %
from keras import backend as K
def rnorm(ip):
    # x[0]=x[0]/K.max(x[1])
    x=ip[0]
    y=ip[1]
    # print(K.max(y))
    # res=x/K.max(y)
    # res=K.abs(x)/K.max(x)
    res=(x-K.min(x))/K.max(K.abs(x))
    # res=(x-K.mean(x)+K.std(x))/K.max(K.abs(x))
    return res

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def c_o4(y_pred,y_true):
    y_true = tf.reshape(y_true, [-1, objects.shape[1], objects.shape[2], objects.shape[3]])  # batch, H, W, 1
    y_pred = tf.reshape(y_pred, [-1, objects.shape[1], objects.shape[2], objects.shape[3]])
    y_pred = y_pred + 1e-6
    y_preds=tf.squeeze(y_pred)
    y_predsn=rnorm([y_preds,y_preds])
    y_trues=tf.squeeze(y_true)
    y_truesn=rnorm([y_trues,y_trues])
    
    # loss0=bce(y_predsn,y_truesn)
    
    y_prede=tf.expand_dims(y_predsn,3)
    y_truee=tf.expand_dims(y_truesn,3)
    y_prede=tf.expand_dims(y_prede,0)
    y_truee=tf.expand_dims(y_truee,0)

    
    y_true1=tf.reduce_sum(y_truee, 1)
    y_true1d=y_true1/tf.reduce_max(y_true1)
    # y_true1=tf.transpose(y_true1,[0,2,1,3])
    # y_true1dr=tf.reshape(y_true1d,[y_true1d.shape[1]*lidn,y_true1d.shape[2],1])
    y_pred1=tf.reduce_sum(y_prede, 1)
    y_pred1d=y_pred1/tf.reduce_max(y_pred1)
    
    y_pred1d=tf.where(tf.math.is_nan(y_pred1d), tf.zeros_like(y_pred1d), y_pred1d)
    y_true1d=tf.where(tf.math.is_nan(y_true1d), tf.zeros_like(y_true1d), y_true1d)
    # y_pred1=tf.transpose(y_pred1,[0,2,1,3])
    # y_pred1dr=tf.reshape(y_pred1d,[y_pred1d.shape[1]*lidn,y_pred1d.shape[2],1])
    
    # print(y_true1d.shape)
    # print(y_pred1d.shape)
    loss1=1-tf.image.ssim(y_true1d,y_pred1d,max_val=1.0, filter_size=7)
    
    # loss1=bce(y_true1d,y_pred1d)
    # loss1=K.mean(K.square(y_true1d - y_pred1d))
    # loss1=loss1/tf.reduce_max(loss1)

    y_true2=tf.reduce_sum(y_truee, 2)
    y_true2d=y_true2/tf.reduce_max(y_true2)
    # y_true2=tf.transpose(y_true2,[0,2,1,3])
    # y_true2dr=tf.reshape(y_true2d,[y_true2d.shape[1]*lidn,y_true2d.shape[2],1])
    y_pred2=tf.reduce_sum(y_prede, 2)
    y_pred2d=y_pred2/tf.reduce_max(y_pred2)
    
    y_pred2d=tf.where(tf.math.is_nan(y_pred2d), tf.zeros_like(y_pred2d), y_pred2d)
    y_true2d=tf.where(tf.math.is_nan(y_true2d), tf.zeros_like(y_true2d), y_true2d)
    # y_pred2=tf.transpose(y_pred2,[0,2,1,3])
    # y_pred2dr=tf.reshape(y_pred2d,[y_pred2d.shape[1]*lidn,y_pred2d.shape[2],1])
    loss2=1-tf.image.ssim(y_true2d,y_pred2d,max_val=1.0, filter_size=7)
    # loss2=bce(y_true2d,y_pred2d)
    # loss2=K.mean(K.square(y_true2d - y_pred2d))
    # loss2=loss2/tf.reduce_max(loss2)

    y_true3=tf.reduce_sum(y_truee, 3)
    y_true3d=y_true3/tf.reduce_max(y_true3)
    # y_true3=tf.transpose(y_true3,[0,2,1,3])
    # y_true3dr=tf.reshape(y_true3d,[y_true3d.shape[1]*lidn,y_true3d.shape[2],1])
    y_pred3=tf.reduce_sum(y_prede, 3)
    y_pred3d=y_pred3/tf.reduce_max(y_pred3)
    
    y_pred3d=tf.where(tf.math.is_nan(y_pred3d), tf.zeros_like(y_pred3d), y_pred3d)
    y_true3d=tf.where(tf.math.is_nan(y_true3d), tf.zeros_like(y_true3d), y_true3d)
    # y_pred3=tf.transpose(y_pred3,[0,2,1,3])
    # y_pred3dr=tf.reshape(y_pred3d,[y_pred3d.shape[1]*lidn,y_pred3d.shape[2],1])
    loss3=1-tf.image.ssim(y_true3d,y_pred3d,max_val=1.0, filter_size=7)
    # loss3=bce(y_true3d,y_pred3d)
    # loss3=K.mean(K.square(y_true3d - y_pred3d))
    
    loss=loss1+loss2+loss3#+loss0
    # loss=tf.math.reduce_mean(loss)
    return loss#+loss4#+lossbce#loss1+loss2
# %% # --- Layers and model ---
forward_layer = SparseVoxelPixelLayer(voxel_coords, pixel_coords, num_voxels, num_pixels, initial_weights=F_weights_init)
backward_layer = SparsePixelVoxelLayer(voxel_coords, pixel_coords, num_voxels, num_pixels, initial_weights=B_weights_init)
model = UnrolledRLWithRegularizer(forward_layer, backward_layer, image_shape=images.shape[1:3], voxel_shape=objects.shape[1:4], num_iter=3)
# %% # --- Model training ---
model.regularizer_B.trainable = True
model.regularizer_F.trainable = True
# model.regularizer_B.trainable = False
# model.regularizer_F.trainable = False

doff = 8 # reserve datasets for testing
model.compile(optimizer='adam', loss=['binary_crossentropy','binary_crossentropy'], loss_weights=[1,1])
# %%
history=model.fit(X_train[0:Y_train.shape[0]-doff], [X_train[0:Y_train.shape[0]-doff],Y_train[0:Y_train.shape[0]-doff]], epochs=40, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
# %% --- Model testing ---
doff = 8
generated_objects=np.zeros((doff,objects.shape[1], objects.shape[2], objects.shape[3]))
generated_images=np.zeros((doff,images.shape[1], images.shape[2]))
pid_base=60
for pid in range(pid_base,objects.shape[0]):
    # pid=X_train.shape[0]-1
    output = model.predict(X_train[pid:pid+1])
    output_image = output[0].reshape(ix, iy)  # average across depth
    output_object = output[1].reshape(ox, oy, oz)  # average across 
    generated_images[pid-pid_base]=output_image
    generated_objects[pid-pid_base]=output_object
generated_images = generated_images.astype('float32')
generated_objects = generated_objects.astype('float32')
# %
# from scipy.io import savemat
# savemat('output_LiLoR_FLFM.mat',{'outputX':generated_images,'outputY':generated_objects,'history':history.history})
# %% # --- Plot test outputs ---
pid=X_train.shape[0]-3
import time
output = model.predict(X_train[pid:pid+1])

output_image = output[0].reshape(ix, iy)  # average across depth
output_object = output[1].reshape(ox, oy, oz)  # average across 
# %
did = 3
# plt.imshow(output_object.astype('float32').sum(axis=2))#,clim=(0,1))
plt.imshow(output_object[:,:,did].astype('float32'),clim=(0,1))
# plt.imshow(output_image.astype('float32'),clim=(0,1))
plt.title("RL-Reconstructed Object")
plt.show()

