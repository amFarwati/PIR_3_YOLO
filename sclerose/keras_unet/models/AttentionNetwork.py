from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D, SpatialDropout2D, UpSampling2D, Input, concatenate, Multiply
import tensorflow as tf


def upsample_conv(inputs, filters, kernel_size, strides, padding, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer='None'):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer)(inputs)

# for attentionUNet, a conv afterUpSampling is necessary to adpat the number of channels
def upsample_simple(inputs,filters, kernel_size, strides, padding, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer='None'):
    x=UpSampling2D(strides)(inputs)
    return  Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding, kernel_regularizer=kernel_regularizer)(x)


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer='None'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)
    #        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def residual_refinement_block(
        inputs,
        use_batch_norm=True,
        dropout=0.0,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer='None'):
    c1 = Conv2D(filters, (1, 1), activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(inputs)
    if use_batch_norm:
        c1 = BatchNormalization()(c1)
    if dropout > 0.0:
        c1 = SpatialDropout2D(rate=dropout, data_format='channels_last')(c1)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c1)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)
    c = Conv2D(filters, (1, 1), activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)

    return tf.nn.relu(c1+c)

def channel_attention_block(
        high_inputs,
        low_inputs,
        use_batch_norm=True,
        dropout=0.0,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer='None'):
    inputs=concatenate([high_inputs, low_inputs], axis=-1)
    c = GlobalAveragePooling2D()(inputs)
    c = tf.reshape(c, [-1, 1, 1, int(c.shape[-1])])
    c = Conv2D(filters, (1, 1), activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)
    c = Conv2D(filters, (1, 1), activation='sigmoid', kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)
    c = Multiply()([c, low_inputs])
   
    c = Conv2D(filters, (1, 1), activation='sigmoid', kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)

    return tf.nn.relu(high_inputs+c)

def cascaded_pyramid_conv_block(
        inputs,
        stage,
        use_batch_norm=True,
        dropout=0.0,
        filters=16,
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer='None'):
    c1 = Conv2D(filters, (1, 1), activation=activation, kernel_initializer=kernel_initializer, padding=padding,
                kernel_regularizer=kernel_regularizer)(inputs)
    if use_batch_norm:
        c1 = BatchNormalization()(c1)
    if dropout > 0.0:
        c1 = SpatialDropout2D(rate=dropout, data_format='channels_last')(c1)
    inp = [0,3,9,15,21,21]
    pas = c1.shape[-1]//stage
    r = c1.shape[-1]%stage
    if stage > 1:
        x=[c1[:,:,:,:pas]]
    else :
        x = c1[:, :, :, :pas]
    for k in range(1,stage):
        if k == stage-1 :
            t = c1[:, :, :, k * pas:(k + 1) * pas+r]
            t = Conv2D(x[0].shape[-1]+r, (inp[k], inp[k]), activation=activation, kernel_initializer=kernel_initializer,
                       padding=padding,
                       kernel_regularizer=kernel_regularizer)(t)
        else :
            t = c1[:, :, :, k * pas:(k + 1) * pas]
            t = Conv2D(x[0].shape[-1], (inp[k], inp[k]), activation=activation, kernel_initializer=kernel_initializer,
                       padding=padding,
                       kernel_regularizer=kernel_regularizer)(t)
        if use_batch_norm:
            t = BatchNormalization()(t)
        if dropout > 0.0:
            t = SpatialDropout2D(rate=dropout, data_format='channels_last')(t)
        x.append(t)
    if stage>1:
        x = concatenate(x, axis=-1)
    x = Conv2D(filters, (1, 1), activation=activation, kernel_initializer=kernel_initializer, padding=padding,
                kernel_regularizer=kernel_regularizer)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    if dropout > 0.0:
        x = SpatialDropout2D(rate=dropout, data_format='channels_last')(x)
    return tf.nn.relu(x + c1)


def attention_unet_seg_decoder(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=4,
        output_activation='softmax',  # 'sigmoid' or 'softmax'
        kernel_regularizer='None'):
    
    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    # make encoder
    down_layers = []
    down_layers_boundary = []
    stage = 5
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                         kernel_regularizer=kernel_regularizer)
        c = cascaded_pyramid_conv_block(inputs=x, stage=stage, filters=filters, use_batch_norm=use_batch_norm,
                                           dropout=dropout, kernel_regularizer=kernel_regularizer)
        c = residual_refinement_block(inputs=c, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                         kernel_regularizer=kernel_regularizer)
        down_layers_boundary.append(c)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer
        stage = stage - 1

    c = down_layers_boundary[0]
    filters_boundary = c.shape[-1]
    for conv in down_layers_boundary[1:]:
        t = c.shape[1] // conv.shape[1]
        conv = upsample(inputs=conv, filters=filters_boundary, kernel_size=(2, 2), strides=(t, t), padding='same', activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        # conv = UpSampling2D((t, t))(conv)
        # conv = Conv2D(filters_boundary, (3, 3), activation=output_activation, padding='same')(conv)

        c = tf.nn.relu(c + conv)
        c = residual_refinement_block(inputs=c, filters=filters_boundary, use_batch_norm=use_batch_norm, dropout=dropout,
                                      kernel_regularizer=kernel_regularizer)

    if x.shape[0] == None:
        one = tf.ones(x.shape[1:])
        one = tf.reshape(one, (-1, one.shape[0], one.shape[1], one.shape[2]))
    else:
        one = tf.ones(x.shape)
    x = GlobalAveragePooling2D()(x)
    x = tf.reshape(x, [-1, 1, 1, int(x.shape[-1])])
    x = UpSampling2D((int(one.shape[1]), int(one.shape[2])))(x)
    # x = Multiply()([x, one])

    c1 = GlobalAveragePooling2D()(down_layers_boundary[-1])
    # c1 = GlobalAveragePooling2D()(c)
    c1 = tf.reshape(c1, [-1, 1, 1, int(c1.shape[-1])])
    c1 = UpSampling2D((int(one.shape[1]), int(one.shape[2])))(c1)
    # c1 = upsample(int(x.shape[-1]), (2, 2), strides=(int(one.shape[1]), int(one.shape[2])), padding='same')(c1)
    # c1 = Multiply()([c1, one])

    x = concatenate([x,c1], axis=-1) #tf.nn.relu(c1 + x)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    stage = 5 - len(down_layers) + 1
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        conv = cascaded_pyramid_conv_block(inputs=conv, stage=stage, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                           kernel_regularizer=kernel_regularizer)
        conv = residual_refinement_block(inputs=conv, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                         kernel_regularizer=kernel_regularizer)
        x = upsample( inputs=x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        x = channel_attention_block(high_inputs=x, low_inputs=conv, filters=filters, use_batch_norm=use_batch_norm,
                                    dropout=dropout, kernel_regularizer=kernel_regularizer)
        x = residual_refinement_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                      kernel_regularizer=kernel_regularizer)
        stage=stage+1

    # x = tf.nn.relu(c + x)
    outputs_seg = Conv2D(num_classes, (1, 1), activation=output_activation, name="OutputSeg")(x)
    outputs_boundary = Conv2D(2, (1, 1), activation=output_activation, name="OutputBoundary")(c)
    output_synergy = concatenate([outputs_seg, outputs_boundary], axis=-1, name="OutputSynergy")

    model = Model(inputs=[inputs], outputs=[outputs_seg, outputs_boundary, output_synergy])
    return model

def attention_unet_smooth_network(
    input_shape,
    num_classes=1,
    use_batch_norm=True, 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.3, 
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    cnn_activation='relu', # 'LeakyReLU', 'elu', 'relu',... 
    output_activation='softmax', # 'sigmoid' or 'softmax'
    kernel_regularizer='None'):  
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    # make encoder
    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, kernel_regularizer=kernel_regularizer)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    # x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
    #                  kernel_regularizer=kernel_regularizer)
    # down_layers.append(x)
    # dropout += dropout_change_per_layer
    # filters = filters * 2
    # x = AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None)(x)

    if x.shape[0]==None:
        one = tf.ones(x.shape[1:])
        one = tf.reshape(one, (-1, one.shape[0], one.shape[1], one.shape[2]))
    else :
        one = tf.ones(x.shape)
        
    x = GlobalAveragePooling2D()(x)
    x = tf.reshape(x, [-1, 1, 1, int(x.shape[-1])])
    x = Multiply()([x, one])

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        conv = residual_refinement_block(inputs=conv, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, kernel_regularizer=kernel_regularizer)
        x = upsample(inputs=x, filters= filters, kernel_size=(2, 2), strides=(2, 2), padding='same',  activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        x = channel_attention_block(high_inputs=x, low_inputs=conv, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, kernel_regularizer=kernel_regularizer)
        x = residual_refinement_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                  kernel_regularizer=kernel_regularizer)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def attention_resunet_seg_decoder(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=4,
        cnn_activation='relu', # 'LeakyReLU', 'elu', 'relu',... 
        output_activation='softmax',  # 'sigmoid' or 'softmax'
        kernel_regularizer='None'):
   
    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    # make encoder
    down_layers = []
    down_layers_boundary = []
    stage = 5
    for l in range(num_layers):
        x = residual_refinement_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                         kernel_regularizer=kernel_regularizer)
        c = cascaded_pyramid_conv_block(inputs=x, stage=stage, filters=filters, use_batch_norm=use_batch_norm,
                                           dropout=dropout,
                                           kernel_regularizer=kernel_regularizer)
        c = residual_refinement_block(inputs=c, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                         kernel_regularizer=kernel_regularizer)
        down_layers_boundary.append(c)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer
        stage = stage - 1

    c = down_layers_boundary[0]
    filters_boundary = c.shape[-1]
    for conv in down_layers_boundary[1:]:

        t = c.shape[1] // conv.shape[1]
        conv = upsample(inputs=conv, filters=filters_boundary, kernel_size=(2, 2), strides=(t, t), padding='same',  activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        # conv = UpSampling2D((t, t))(conv)
        # conv = Conv2D(filters_boundary, (3, 3), activation=output_activation, padding='same')(conv)

        c = tf.nn.relu(c + conv)
        c = residual_refinement_block(inputs=c, filters=filters_boundary, use_batch_norm=use_batch_norm, dropout=dropout,
                                      kernel_regularizer=kernel_regularizer)

    if x.shape[0] == None:
        one = tf.ones(x.shape[1:])
        one = tf.reshape(one, (-1, one.shape[0], one.shape[1], one.shape[2]))
    else:
        one = tf.ones(x.shape)
    # if resid_1==True:
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = tf.reshape(x, [-1, 1, 1, int(x.shape[-1])])
    #     x = UpSampling2D((int(one.shape[1]), int(one.shape[2])))(x)
    #     # x = Multiply()([x, one])
    #
    #     c1 = GlobalAveragePooling2D()(down_layers_boundary[-1])
    #     # c1 = GlobalAveragePooling2D()(c)
    #     c1 = tf.reshape(c1, [-1, 1, 1, int(c1.shape[-1])])
    #     c1 = UpSampling2D((int(one.shape[1]), int(one.shape[2])))(c1)
    #     # c1 = upsample(int(x.shape[-1]), (2, 2), strides=(int(one.shape[1]), int(one.shape[2])), padding='same')(c1)
    #     # c1 = Multiply()([c1, one])
    #
    #     x=tf.nn.relu(c1 + x)

    # if concat_1==True:

    x = GlobalAveragePooling2D()(x)
    c1 = GlobalAveragePooling2D()(down_layers_boundary[-1])
    x = concatenate([x,c1], axis=-1)
    x = tf.reshape(x, [-1, 1, 1, int(x.shape[-1])])
    x = upsample(inputs=x, filters=int(x.shape[-1])//2, kernel_size=(2, 2), strides=(int(one.shape[1]), int(one.shape[2])), padding='same')

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    stage = 5 - len(down_layers) + 1
    up_layers = []
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        conv = cascaded_pyramid_conv_block(inputs=conv, stage=stage, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                           kernel_regularizer=kernel_regularizer)
        conv = residual_refinement_block(inputs=conv, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                         kernel_regularizer=kernel_regularizer)
        x = upsample(inputs=x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same',  activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        x = channel_attention_block(high_inputs=x, low_inputs=conv, filters=filters, use_batch_norm=use_batch_norm,
                                    dropout=dropout, kernel_regularizer=kernel_regularizer)
        x = residual_refinement_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,
                                      kernel_regularizer=kernel_regularizer)
        up_layers.append(x)
        stage=stage+1

    c2=up_layers[-1]
    up_layers_upsample=[]
    for conv in up_layers[:-1]:
        t = c2.shape[1] // conv.shape[1]
        # if concat_2==True:
        conv = upsample(inputs=conv, filters=conv.shape[-1], kernel_size=(2, 2), strides=(t, t), padding='same', activation=cnn_activation, kernel_regularizer=kernel_regularizer)
        # if resid_2==True:
        #     conv = upsample(filters_boundary, (2, 2), strides=(t, t), padding='same')(conv) #conv.shape[-1]
        up_layers_upsample.append(conv)
    # if concat_2 == True:
    x = concatenate(up_layers_upsample, axis=-1)
    # if resid_2 == True:
    #     res=up_layers_upsample[0]
    #     for i in up_layers_upsample[1:]:
    #         res = res + i
    #     x = tf.nn.relu(up_layers_upsample + x)
    outputs_seg = Conv2D(num_classes, (1, 1), activation=output_activation, name="OutputSeg")(x)
    outputs_boundary = Conv2D(2, (1, 1), activation='sigmoid', name="OutputBoundary")(c)
    output_synergy = concatenate([outputs_seg, outputs_boundary], axis=-1, name="OutputSynergy")

    model = Model(inputs=[inputs], outputs=[outputs_seg, outputs_boundary, output_synergy])
    return model