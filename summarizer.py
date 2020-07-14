import tensorflow as tf

model = tf.keras.models.load_model('saved_model\d_256\E_1000')
model.get_config()