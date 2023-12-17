from math import sqrt
from einops import rearrange
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv3D, MaxPooling3D, BatchNormalization, Add, ZeroPadding2D

# learning_rate = [1e-4]
learning_rate = [2.5e-4]  #1e-3
#weight_decay = 0.0001
batch_size = 128 #128
num_epochs = 100
image_size = [25, 25]  # We'll resize input images to this size
patch_size = 5  # Size of the patches to be extract from the input images
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
num_heads = 6 #8
transformer_layers = 3 #8
mlp_head_units = [1024, 512]  # Size of the dense layers of the final classifier
input_shape = (25, 25, 128)
num_classes= 3

emb_dims = [64, 192, 384]
emb_kernel = [3,3,2]
emb_stride = [2,2,2]
trans_depth = [2,2,2]
projection_dim = [64, 192, 384]

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def conv_block(input):
  zp = ZeroPadding2D(padding=(1, 1), input_shape=(62,250,5))(input)
  conv_1 = Conv2D(filters=128, kernel_size=(15,3), padding='valid', strides=(1,5))(zp)
  mp = MaxPooling2D(pool_size=2, strides=2)(conv_1)
  drpot = Dropout(0.2)(mp)
  return drpot

def create_vit_classifier():
    inputs = layers.Input(shape= (62, 265, 5))
    features = conv_block(inputs)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(features)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim[0])(patches)
    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
      for j in range(trans_depth[i]):
        if j==0:
          img_dim = int(sqrt(encoded_patches.shape[1]))
          r1 = rearrange(encoded_patches, 'b (h w) c -> b h w c', b=encoded_patches.shape[0], h=img_dim, w=img_dim, c=encoded_patches.shape[2])
          zp_trans = ZeroPadding2D(padding=emb_kernel[i]//2)(r1)
          cnv = Conv2D(filters=emb_dims[i], kernel_size=emb_kernel[i], strides=emb_stride[i])(zp_trans)
          r2 = rearrange(cnv, 'b h w c -> b (h w) c', b=cnv.shape[0], h=cnv.shape[1], w=cnv.shape[2], c=cnv.shape[3])
          x1 = layers.LayerNormalization(epsilon=1e-6)(r2)
        else:
          x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Layer normalization 1.
        # x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim[0], dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x1])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        transformer_units = [
          projection_dim[0] * 2,
          projection_dim[i],
        ]
        if j==(trans_depth[i]-1):
          x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    outputs = layers.Softmax()(logits)
    # Create the Keras model.
    model_vit = keras.Model(inputs=inputs, outputs=outputs)
    return model_vit

def run_experiment(model_vit, train_data, train_labels_reshaped, test_data, test_labels_reshaped):

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    reduce_lr_exp = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=4, min_lr=1e-6)  

    history = model_vit.fit(
        x= train_data,
        y= train_labels_reshaped,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback, reduce_lr_exp],
    )

    model_vit.load_weights(checkpoint_filepath)
    _, accuracy = model_vit.evaluate(test_data, test_labels_reshaped)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history