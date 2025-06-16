import tensorflow as tf
from keras.saving import register_keras_serializable

# Register the custom PSNR function
@register_keras_serializable()
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)

# Register the custom SSIM function
@register_keras_serializable()
def SSIM(y_true, y_pred):
    max_pixel = 1.0
    return tf.image.ssim(y_true, y_pred, max_val=max_pixel)

# Load your Keras model with custom_objects
model = tf.keras.models.load_model('autoencoder.keras', custom_objects={'PSNR': PSNR, 'SSIM': SSIM})

# Create a TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Specify bfloat16 supported type
converter.target_spec.supported_types = [tf.bfloat16]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('model_bf16.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted and saved as 'model_bf16.tflite'")
