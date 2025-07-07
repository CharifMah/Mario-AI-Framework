import tensorflow as tf
model = tf.saved_model.load("mario_lstm_savedmodel")
infer = model.signatures["serving_default"]
import numpy as np
x = np.zeros((1, 100, 1), dtype=np.float32)  # ou un vrai seed si tu veux
result = infer(tf.convert_to_tensor(x))
print(result)
