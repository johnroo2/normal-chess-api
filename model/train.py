import tensorflow as tf
from preprocess import preprocess_from, preprocess_to
from model import model

MODEL_TRAIN_FROM = False

BATCH_SIZE = 1024
VAL_SIZE = BATCH_SIZE * 10
EPOCHS = 2
STEPS_PER_EPOCH = 25

ds = tf.data.TextLineDataset('.venv/model/data/moves.txt').repeat()
ds = ds.shuffle(buffer_size=1000)
ds = ds.map(preprocess_from if MODEL_TRAIN_FROM else preprocess_to, num_parallel_calls=tf.data.AUTOTUNE)

train = ds.skip(VAL_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val = ds.take(VAL_SIZE).batch(BATCH_SIZE)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    '.venv/model/archive/{}/model{}.h5'.format("from" if MODEL_TRAIN_FROM else "to", "{epoch:02d}"),
    save_best_only=False, 
    save_weights_only=False, 
    verbose=1,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='.venv/model/logs/callback',
    profile_batch='10,14',
)

model.fit(
    train,
    epochs=EPOCHS, 
    steps_per_epoch=STEPS_PER_EPOCH, 
    validation_data=val,
    callbacks=[tensorboard_callback, checkpoint_callback], 
    verbose=1,
)