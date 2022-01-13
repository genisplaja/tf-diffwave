import tensorflow as tf
from tensorflow.keras import layers
from .unet_config import config as unet_config

import librosa


class DilatedConv1d(tf.keras.layers.Layer):
    """Custom implementation of dilated convolution 1D 
    because of the issue https://github.com/tensorflow/tensorflow/issues/26797.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_rate):
        """Initializer.
        Args:
            in_channels: int, input channels.
            out_channels: int, output channels.
            kernel_size: int, size of the kernel.
            dilation_rate: int, dilation rate.
        """
        super(DilatedConv1d, self).__init__()
        self.dilations = dilation_rate

        init = tf.keras.initializers.GlorotUniform()
        self.kernel = tf.Variable(
            init([kernel_size, in_channels, out_channels], dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            tf.zeros([1, 1, out_channels], dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        """Pass to dilated convolution 1d.
        Args:
            inputs: tf.Tensor, [B, T, Cin], input tensor.
        Returns:
            outputs: tf.Tensor, [B, T', Cout], output tensor.
        """
        conv = tf.nn.conv1d(
            inputs, self.kernel, 1, padding='SAME', dilations=self.dilations)
        return conv + self.bias


class UnetConvBlock(tf.keras.Model):
    def __init__(self, n_filters, initializer, activation, kernel_size=(5, 5), strides=(2, 2), padding='same'):
        super(UnetConvBlock, self).__init__()
        self.n_filters=n_filters
        self.initializer=initializer
        self.activation=activation
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding

        self.conv2d = layers.Conv2D(
            filters=n_filters,
            kernel_size=(5, 5),
            padding=padding,
            strides=(2, 2),
            kernel_initializer=initializer)
        self.batch_norm_encoder = layers.BatchNormalization(momentum=0.9, scale=True)
        self.activation_encoder = self._get_activation(activation)

    def call(self, inputs):
        x = inputs
        x = self.conv2d(x)
        x = self.batch_norm_encoder(x)
        x = self.activation_encoder(x)
        return x

    @staticmethod
    def _get_activation(name):
        if name == 'leaky_relu':
            return layers.LeakyReLU(alpha=0.2)
        return tf.keras.layers.Activation(name)


class UnetUpconvBlock(tf.keras.Model):
    def __init__(self, n_filters, initializer, activation, dropout, skip, kernel_size=(5, 5), strides=(2, 2), padding='same'):
        super(UnetUpconvBlock, self).__init__()
        self.n_filters=n_filters
        self.initializer=initializer
        self.activation=activation
        self.dropout=dropout
        self.skip=skip
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding

        if self.skip:
             self.concatenate_decoder = layers.Concatenate(axis=3)
        self.deconv = layers.Conv2DTranspose(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            kernel_initializer=initializer)
        self.batch_norm = layers.BatchNormalization(momentum=0.9, scale=True)
        if self.dropout:
            self.dropout_decoder = layers.Dropout(0.5)
        self.activation_decoder = self._get_activation(activation)

    def call(self, x, x_encoder):
        if self.skip:
            x = self.concatenate_decoder([x, x_encoder])
        x = self.deconv(x)
        x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout_decoder(x)
        x = self.activation_decoder(x)
        return x

    @staticmethod
    def _get_activation(name):
        if name == 'leaky_relu':
            return layers.LeakyReLU(alpha=0.2)
        return tf.keras.layers.Activation(name)


class Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, channels, kernel_size, dilation, last=False):
        """Initializer.
        Args:
            channels: int, basic channel size.
            kernel_size: int, kernel size of the dilated convolution.
            dilation: int, dilation rate.
            last: bool, last block or not.
        """
        super(Block, self).__init__()
        self.channels = channels
        self.last = last

        self.proj_embed = tf.keras.layers.Dense(channels)
        self.conv = DilatedConv1d(
            channels, channels * 2, kernel_size, dilation)
        self.proj_mel = tf.keras.layers.Conv1D(channels * 2, 1)

        if not last:
            self.proj_res = tf.keras.layers.Conv1D(channels, 1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, 1)

    def call(self, inputs, embedding, mel):
        """Pass wavenet block.
        Args:
            inputs: tf.Tensor, [B, T, C(=channels)], input tensor.
            embedding: tf.Tensor, [B, E], embedding tensor for noise schedules.
            mel: tf.Tensor, [B, T // hop, M], mel-spectrogram conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        # [B, C]
        embedding = self.proj_embed(embedding)
        # [B, T, C]
        x = inputs + embedding[:, None]
        # [B, T, Cx2]
        x = self.conv(x) + self.proj_mel(mel)
        # [B, T, C]
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate
        # [B, T, C]
        residual = (self.proj_res(x) + inputs) / 2 ** 0.5 if not self.last else None
        skip = self.proj_skip(x)
        return residual, skip


class WaveNet(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(WaveNet, self).__init__()
        self.config = config
        # signal proj
        self.proj = tf.keras.layers.Conv1D(config.channels, 1)
        # embedding
        self.embed = self.embedding(config.iter)
        self.proj_embed = [
            tf.keras.layers.Dense(config.embedding_proj)
            for _ in range(config.embedding_layers)]

        self.num_unet_layers = unet_config.N_LAYERS
        self.unet_encoder = []
        unet_initializer = tf.random_normal_initializer(stddev=0.02)
        # Encoder
        encoder_filters = []
        for i in range(self.num_unet_layers):
            n_filters = unet_config.FILTERS_LAYER_1 * (2 ** i)
            encoder_filters.append(n_filters)
            self.unet_encoder.append(
                UnetConvBlock(
                    n_filters,
                    unet_initializer,
                    unet_config.ACTIVATION_ENCODER))
        # Decoder
        self.unet_decoder = []
        decoder_filters = list(reversed(encoder_filters[:-1]))
        for i in range(self.num_unet_layers):
            # not dropout in the first block and the last two encoder blocks
            dropout = True if i in unet_config.BLOCKS_DROPOUT else False
            # the last layer is different
            is_final_block = True if i == self.num_unet_layers - 1 else False
            # not skip in the first encoder block - the deepest
            skip = False if i == 0 else True
            if is_final_block:
                n_filters = 1
                activation = unet_config.ACT_LAST
            else:
                n_filters = decoder_filters[i]
                activation = unet_config.ACTIVATION_DECODER
            self.unet_decoder.append(
                UnetUpconvBlock(
                    n_filters,
                    unet_initializer,
                    activation,
                    dropout,
                    skip))
        # mel-upsampler
        self.upsample = [
            tf.keras.layers.Conv2DTranspose(
                1,
                config.upsample_kernel,
                config.upsample_stride,
                padding='same')
            for _ in range(config.upsample_layers)]
        # wavenet blocks
        self.blocks = []
        layers_per_cycle = config.num_layers // config.num_cycles
        for i in range(config.num_layers):
            dilation = config.dilation_rate ** (i % layers_per_cycle)
            self.blocks.append(
                Block(
                    config.channels,
                    config.kernel_size,
                    dilation,
                    last=i == config.num_layers - 1))  
        # for output
        self.proj_out = [
            tf.keras.layers.Conv1D(config.channels, 1, activation=tf.nn.relu),
            tf.keras.layers.Conv1D(1, 1)]

        # convert from spec to melfilt
        melfilter = librosa.filters.mel(
            config.sr, config.fft, config.mel, config.fmin, config.fmax).T
        self.melfilter = tf.convert_to_tensor(melfilter[:-1, :])

    def call(self, signal, timestep, spec, eval=False):
        """Generate output signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], int, timesteps of current markov chain.
            spec: tf.Tensor, TODO
        Returns:
            tf.Tensor, [B, T], generated.
        """
        # [B, T, C(=channels)]
        x = tf.nn.relu(self.proj(signal[..., None]))
        # [B, E']
        embed = tf.gather(self.embed, timestep - 1)
        # [B, E]
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))
        # [B, T, M, 1], treat as 2D tensor.

        # passing spec through unet network...
        # [B, fft, T // hop]
        #print('spec', spec.shape)
        spec = spec[..., None]
        estimation = spec
        encoder_tmp = []
        for encoder_layer in self.unet_encoder:
            estimation = encoder_layer(estimation)
            encoder_tmp.append(estimation)
        for encoder_layer, decoder_layer in zip(reversed(encoder_tmp), self.unet_decoder):
            estimation = decoder_layer(estimation, encoder_layer)
        estimation =  tf.squeeze(layers.multiply([estimation, spec]), axis=-1)
        #print('estimation', estimation.shape)
        # [B, T // hop, ftt // 2]
        estimation = tf.transpose(estimation, [0, 2, 1])
        #print('estimation trans', estimation.shape)
        # [B, T // hop, mel], [fft // 2, mel]
        mel = estimation @ self.melfilter
        #print('mel', mel.shape)
        # [B, T // hop, mel]
        mel = tf.math.log(tf.maximum(mel, self.config.eps))
        #print('mel log', mel.shape)
        # Add dimension
        if eval is True:
            mel = tf.reshape(mel, [1, mel.shape[0]*mel.shape[1], self.config.mel])
        mel = mel[..., None]
        #print('mel add dim', mel.shape)
        for upsample in self.upsample:
            mel = tf.nn.leaky_relu(upsample(mel), self.config.leak)
        # [B, T, M]
        mel = tf.squeeze(mel, axis=-1)

        context = []
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            x, skip = block(x, embed, mel)
            context.append(skip)
        # [B, T, C]
        scale = self.config.num_layers ** 0.5
        context = tf.reduce_sum(context, axis=0) / scale
        # [B, T, 1]
        for proj in self.proj_out:
            context = proj(context)
        # [B, T]
        return tf.squeeze(context, axis=-1)

    def embedding(self, iter):
        """Generate embedding.
        Args:
            iter: int, maximum iteration.
        Returns:
            tf.Tensor, [iter, E(=embedding_size)], embedding vectors.
        """
        # [E // 2]
        logit = tf.linspace(0., 1., self.config.embedding_size // 2)
        exp = tf.pow(10, logit * self.config.embedding_factor)
        # [iter]
        timestep = tf.range(1, iter + 1)
        # [iter, E // 2]
        comp = exp[None] * tf.cast(timestep[:, None], tf.float32)
        # [iter, E]
        return tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)
