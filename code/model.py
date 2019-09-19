from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


def create_model(type, img_shape, n_hidden, dropout, label):

    pretrained = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_shape[1], img_shape[2], img_shape[3]),
    )

    network = layers.Flatten()(pretrained.output)
    network = layers.Dense(n_hidden, activation="relu")(network)
    network = layers.Dropout(dropout)(network)

    if label == "fine":
        out = layers.Dense(100, activation="softmax")(network)
    elif label == "coarse":
        out = layers.Dense(20, activation="softmax")(network)
    else:
        out_fine = layers.Dense(100, activation="softmax", name="fine")(network)
        out_coarse = layers.Dense(20, activation="softmax", name="coarse")(network)
        out = (out_fine, out_coarse)

    input_image = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))

    model = Model(inputs=pretrained.input, outputs=out)
    print(model.summary())

    return model
