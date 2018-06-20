from keras.models import Model, Sequential
from keras.layers import add, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
# from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import utils


def compile_model(model, soft_sigmoid=False):
    lw = [1.0, 1.0, 1.0]
    if soft_sigmoid is False:
        # optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    else:
        model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'], loss_weights=lw)
    return model


def flow_create_model(classes, model_name, soft_sigmoid=False, image_shape=(224, 224), opt_flow_len=20):
    conv_fusion = False
    freeze_all = True
    input_shape = (image_shape[0], image_shape[1], opt_flow_len)  # OF has 2 channels
    if model_name == "resnet50":
        base_model = resnet50_of(input_shape, pooling=None)  # resnet is bae <3
    elif model_name == "vgg16":
        base_model = vgg16_of(input_shape, pooling=None)  # VGG has a more straightforward, architecture but worse acc and way more params
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    if conv_fusion is True:
        x = Conv2D(64, (2, 2))(x)
    # add a fully-connected layer with 1024 (should change)
    x = Dense(1024, activation='relu')(x)

    # Freeze layers
    if freeze_all is True:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        if model_name == "resnet50":
            # print base_model.summary
            for layer in base_model.layers[:172]:
                layer.trainable = False
            for layer in base_model.layers[172:]:
                layer.trainable = True
        elif model_name == "shallow":
            pass

    if soft_sigmoid is False:
        predictions = Dense(len(classes), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    else:
        pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(x)
        pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(x)
        pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(x)

        model = Model(inputs=base_model.input, outputs=[pred_pose, pred_obj_human, pred_human_human])
    return model


def vgg16_of(input_shape, pooling='avg'):
    img_input = Input(shape=input_shape)
    # using matconvnet naming conventions
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Top is not included!
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name='vgg16')
    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_of(input_shape, pooling='avg'):
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=(3, 3), name='pad1')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # Top is not included!
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name='resnet50')
    return model


def Shallow(include_top=False, input_shape=(224, 224)):
    # Architecture from the original paper of 2stream
    # http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
    model = Sequential()

    model.add(Conv2D(96, (7, 7), strides=2, padding='same', name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # NOTE This is the only different layer between RGB and OF streams in the original paper
    model.add(Conv2D(256, (5, 5), strides=2, padding='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv3'))
    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv4'))
    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # full6
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.9))

    # full7
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(1024, activation='relu'))

    return model
