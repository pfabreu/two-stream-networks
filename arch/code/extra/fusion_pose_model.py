from keras.layers import concatenate, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from rgb_model import rgb_create_model
from flow_model import flow_create_model
from pose_model import pose_create_model
import sys
import utils


def prepare_rgb_stream(classes, rgb_weights, model_name):
    original_rgb_stream, k = rgb_create_model(classes, soft_sigmoid=True, model_name=model_name, freeze_all=True, conv_fusion=False)
    # print(original_rgb_stream.summary())
    if rgb_weights is None:
        print("No saved rgb_weights weights file, please use fusion weights!")
        sys.exit(0)
    else:
        original_rgb_stream.load_weights(rgb_weights)

        for layer in original_rgb_stream.layers:
            layer.trainable = False
        return original_rgb_stream


def prepare_flow_stream(classes, flow_weights, model_name):
    original_flow_stream, k = flow_create_model(classes, model_name=model_name, soft_sigmoid=True, image_shape=(224, 224), opt_flow_len=20, freeze_all=True, conv_fusion=False)
    # print(original_flow_stream.summary())
    if flow_weights is None:
        print("Aborting, No saved flow_weights weights file, please use fusion weights!")
        sys.exit(0)
    else:
        original_flow_stream.load_weights(flow_weights)

        for layer in original_flow_stream.layers:
            # Change layer names (add an of for the optical flow layers)
            layer.name = layer.name + "of"
            layer.trainable = False
        return original_flow_stream


def prepare_pose_stream(classes, pose_weights):
    original_pose_stream = pose_create_model(classes, soft_sigmoid=True, image_shape=(300, 300), model_name='alexnet')
    if pose_weights is None:
        print("Aborting, No saved pose_weights weights file, please use fusion weights!")
        sys.exit(1)
    else:
        original_pose_stream.load_weights(pose_weights)
        for layer in original_pose_stream.layers:
            # Change layer names (add an of for the optical flow layers)
            layer.name = layer.name + "pose"
            layer.trainable = False
        return original_pose_stream


class FusionPoseModel():

    def __init__(self, classes, rgb_weights, flow_weights, pose_weights):
        # Simple non-time-distributed model
        rgb_m = prepare_rgb_stream(classes, rgb_weights, "resnet50")
        rgb_m.layers.pop()
        rgb_m.layers.pop()
        rgb_m.layers.pop()
        flow_m = prepare_flow_stream(classes, flow_weights, "resnet50")
        flow_m.layers.pop()
        flow_m.layers.pop()
        flow_m.layers.pop()
        pose_m = prepare_pose_stream(classes, pose_weights)
        pose_m.layers.pop()
        pose_m.layers.pop()
        pose_m.layers.pop()
        # TODO Maybe not so many pops
        # print(flow_m.summary())
        # print(rgb_m.summary())
        print(pose_m.summary())
        # Since mode is concat the next conv layer after will learn rgb+flow filters
        m = concatenate([rgb_m.layers[-1].output, flow_m.layers[-1].output, pose_m.layers[-1].output, ], axis=-1)
        # Convolutional fusion layer
        # x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(m)
        # Dense fusion layer
        x = Dense(1024, activation='relu')(m)
        # Average pooling
        # x = AveragePooling2D()(x)
        # Add some Dropout layers?
        x = Dropout(0.5)(x)
        # Add fully connected (merge any 1D inputs here)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        # Add final sigmoid/softsigmoid layer
        pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(x)
        pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(x)
        pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(x)

        self.model = Model(inputs=[rgb_m.input, flow_m.input, pose_m.input], outputs=[pred_pose, pred_obj_human, pred_human_human])

    def compile_model(self, soft_sigmoid=False):
        if soft_sigmoid is False:
            self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(optimizer=Adam(), loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'], loss_weights=[1.0, 1.0, 1.0])
