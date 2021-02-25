import tensorflow as tf

def get_model(lr):
    input1 = tf.keras.layers.Input(4)
    hidden1 = tf.keras.layers.Dense(128, "relu")(input1)
    output = tf.keras.layers.Dense(1, "sigmoid")(hidden1)
    model = tf.keras.models.Model(inputs=[input1], outputs=[output])



    return model