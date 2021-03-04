import tensorflow as tf

def get_model():
    input1 = tf.keras.layers.Input(4)
    X = tf.keras.layers.Dense(128, "relu")(input1)
    X = tf.keras.layers.Dense(256, "relu")(X)
    X = tf.keras.layers.Dropout(.2)(X)
    X = tf.keras.layers.Dense(256, "relu")(X)
    X = tf.keras.layers.Dropout(.2)(X)
    X = tf.keras.layers.Dense(128, "relu")(X)
    output = tf.keras.layers.Dense(1, "sigmoid")(X)
    model = tf.keras.models.Model(inputs=[input1], outputs=[output])



    return model