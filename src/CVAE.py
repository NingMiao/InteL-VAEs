import tensorflow as tf

#Default params
latent_dim=100
encoder_params=[]
encoder_params.append([32, 4, (2,2)]) #Filter, kernelsize, stride
encoder_params.append([64, 4, (2,2)])
encoder_params.append([128, 4, (2,2)])
decoder_params=[]
decoder_params.append([128, 4, (2,2)])
decoder_params.append([64, 4, (2,2)])
decoder_params.append([32, 4, (2,2)])
image_shape=[28, 28, 1]




class CVAE(tf.keras.Model):
  def __init__(self, latent_dim=latent_dim, encoder_params=encoder_params, decoder_params=decoder_params, image_shape=image_shape):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.image_shape=image_shape
    #Inference Net
    self.inference_net = []
    self.inference_net_batch_normalization_list=[]
    for param in encoder_params:
        self.inference_net.append(tf.keras.layers.Conv2D(
              filters=param[0], kernel_size=param[1], strides=param[2], padding='SAME', activation=None))
        self.inference_net.append(tf.keras.layers.BatchNormalization())
        self.inference_net_batch_normalization_list.append(len(self.inference_net)-1)
        self.inference_net.append(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    self.inference_net.append(tf.keras.layers.Flatten())
    self.inference_net.append(tf.keras.layers.Dense(units=latent_dim*2, activation=None))

    
    #Generation Net
    image_x_size=image_shape[0]
    image_y_size=image_shape[1]
    image_dim=image_shape[2]
    
    image_x_size_multi=1
    image_y_size_multi=1
    for param in decoder_params:
        image_x_size_multi*=param[2][0]
        image_y_size_multi*=param[2][1]
    image_x_size_first_layer=int(image_x_size/image_x_size_multi)
    image_y_size_first_layer=int(image_y_size/image_y_size_multi)
    dense_layer_output_size=image_x_size_first_layer*image_y_size_first_layer*decoder_params[0][0]
    
    self.generative_net = []
    self.generative_net_batch_normalization_list=[]
    self.generative_net.append(tf.keras.layers.Dense(units=dense_layer_output_size, activation=tf.nn.relu))
    self.generative_net.append(tf.keras.layers.Reshape(target_shape=[image_x_size_first_layer, image_y_size_first_layer, decoder_params[0][0]]))
    for i in range(len(decoder_params)):
        self.generative_net.append(tf.keras.layers.BatchNormalization())
        self.generative_net_batch_normalization_list.append(len(self.generative_net)-1)
        self.generative_net.append(tf.keras.layers.ReLU())
        if i==len(decoder_params)-1:
            filter_size=image_shape[2]
        else:
            filter_size=decoder_params[i+1][0]
        self.generative_net.append(tf.keras.layers.Conv2DTranspose(
              filters=filter_size,
              kernel_size=decoder_params[i][1],
              strides=decoder_params[i][2],
              padding="SAME",
              activation=None))

  #@tf.function
  #def sample(self, eps=None):
  #  if eps is None:
  #    eps = tf.random.normal(shape=(100, self.latent_dim))
  #  samples=self.decode(eps, apply_sigmoid=True, training=False, before_mapping=True)[0]
  #  return samples
  

  def encode(self, x, training=False):
    layer_output=x
    for i in range(len(self.inference_net)):
        if i in self.inference_net_batch_normalization_list:
            layer_output=self.inference_net[i](layer_output, training=training)
        else:
            layer_output=self.inference_net[i](layer_output)
            
    mean, logvar = tf.split(layer_output, num_or_size_splits=2, axis=1)
    return mean, logvar
    
  def decode(self, z, apply_sigmoid=False, training=False):
    if True:
      layer_output=z
      for i in range(len(self.generative_net)):
        if i in self.generative_net_batch_normalization_list:
            layer_output=self.generative_net[i](layer_output, training=training)
        else:
            layer_output=self.generative_net[i](layer_output)
      logits = layer_output
      if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
      return logits

if __name__=='__main__':
    model=CVAE()
    #CVAE doesn't contain mapping layer