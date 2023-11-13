import tensorflow as tf 

WEIGHTS_TO_FROM = True

model = tf.keras.models.load_model('.venv/model/archive/{}/model{}.h5'.format("from" if WEIGHTS_TO_FROM else "to", f"{(3):02d}"), compile=False)
model.save_weights('.venv/model/archive/weights/{}/checkpoint.ckpt'.format("from" if WEIGHTS_TO_FROM else "to"))
print("Checkpoint Saved!")