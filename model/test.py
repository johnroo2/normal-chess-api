import tensorflow as tf
import numpy as np
from preprocess import fen_to_matrix, invert_fen

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0'
col = fen.split()[1]

if col == 'w':
	a = fen_to_matrix(fen, reshape=True)
else:
	fen = invert_fen(fen)
	a = fen_to_matrix(fen, reshape=True)

model_from = tf.keras.models.load_model('/model/archive/from/model{}.h5'.format(f"{(5):02d}"))
model_to = tf.keras.models.load_model('/model/archive/to/model{}.h5'.format(f"{(5):02d}"))
pf = model_from.predict([a,]).reshape((8, 8))

if col == 'b': print(np.argmax(np.flip(pf, axis=0)))
else: print(np.argmax(pf))

pt = model_to.predict([a,]).reshape((8, 8))

if col == 'b': print(np.argmax(np.flip(pt, axis=0)))
else: print(np.argmax(pt))