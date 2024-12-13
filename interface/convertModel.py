import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("../trained_plant_disease_model.keras")
def convert_to_tflite(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('trained_plant_disease_model.tflite', 'wb') as f:
        f.write(tflite_model)

# Call the conversion function
convert_to_tflite(model)
