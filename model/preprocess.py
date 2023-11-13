import tensorflow as tf
import numpy as np

piece_dict = {
    'p' : [0,1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0,0],
}

def fen_to_matrix(fen, piece_dict=piece_dict, reshape=False):
	fen = fen.split()[0]
	mat = []
	rows = fen.split('/')
	for row in rows:
		a = [] 
		for ch in str(row):
			if ch.isdigit():
				for _ in range(int(ch)):
					a.append(piece_dict['.'])
			else:
				try:	
					a.append(piece_dict[ch])
				except:
					pass
		mat.append(a)
	mat = np.array(mat)
	if reshape:
		mat = np.reshape(mat, (1,) + mat.shape)
	return mat

def invert_fen(fen):
	rows = fen.split()[0].split('/')
	rows.reverse()
	for i in range(8):
		rows[i] = rows[i].swapcase()
	return '/'.join(rows)

def number_to_board(num):
    pos = np.zeros((8, 8))
    pos[num // 8 ][num % 8] = 1
    return pos


def preprocess(inp, is_from):
    def helper_from(inp):
        moved_from_board = []
        fen, moved_from, moved_to = inp.numpy().decode().split('  ')
        moved_from, moved_to = int(moved_from), int(moved_to) 

        fen = fen.split()
        col,fen = fen[1],fen[0]

        if col == 'b':
            fen = invert_fen(fen)
            moved_from,moved_to = 63 - moved_from, 63 - moved_to
            
        moved_from_board = number_to_board(num=moved_from)
        board = fen_to_matrix(fen=fen)

        if col == 'b':
            moved_from_board = np.flip(moved_from_board, axis=1)
                
        return board, moved_from_board
    
    def helper_to(inp):
        moved_to_board = []
        fen, moved_from, moved_to = inp.numpy().decode().split('  ')
        moved_from, moved_to = int(moved_from), int(moved_to) 

        fen = fen.split()
        col,fen = fen[1],fen[0]

        if col == 'b':
            fen = invert_fen(fen)
            moved_from,moved_to = 63 - moved_from, 63 - moved_to
        
        moved_to_board = number_to_board(num=moved_to)
        board = fen_to_matrix(fen=fen)

        if col == 'b':
            moved_to_board = np.flip(moved_to_board, axis=1)
                
        return board, moved_to_board
        
    return tf.py_function(func=helper_from if is_from else helper_to, inp=[inp], Tout=[tf.int32, tf.int32])

def preprocess_from(inp):
    return preprocess(inp, True)

def preprocess_to(inp):
    return preprocess(inp, False)