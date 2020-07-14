import tensorflow as tf

model = tf.saved_model.load("saved_model\d_1024\\b_128\E_440")


concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 64])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
