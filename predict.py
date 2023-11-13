import random
import tensorflow as tf
import numpy as np
from model.preprocess import fen_to_matrix, invert_fen
import chess

from_model = tf.keras.models.load_model('finals/from.h5', compile=False)
to_model = tf.keras.models.load_model('finals/to.h5', compile=False)

debug_piece_dict = {'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
			'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
			'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
			'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
			'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
			'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
			'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
			'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
			'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
			'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
			'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
			'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
			'.' : [0,0,0,0,0,0,0,0,0,0,0,0]}

convert_dict = {
    -10: 'k', 10: 'K', -9: 'q', 9: 'Q',
    -5: 'r', 5: 'R', -3: 'b', 3: 'B',
    -2: 'n', 2: 'N', -1: 'p', 1: 'P',
    0: '1'
}

def merge(input_str):
    result, count = [], 0

    for char in input_str:
        if char == '1': count += 1
        else:
            if count > 1: result.append(str(count))
            elif count == 1: result.append('1')
            result.append(char)
            count = 0

    if(count >= 1): result.append(str(count))
    return ''.join(result)

def convert(board, col, castles, passant, halfmoves, fullmoves):
    rows = []
    for row in board:
        converted_row = ""
        for square in row:
            converted_row += convert_dict[square]
            
        rows.append(merge(converted_row))
    return f"{'/'.join(rows)} {col} {castles} {passant} {halfmoves} {fullmoves}"

def uci_to_row_col(uci):
    parse_square = lambda sq:(7 - (int(sq[1]) - 1), ord(sq[0]) - ord('a'))
    return parse_square(uci[: 2]) + parse_square(uci[2 :])

class Computer():
    def __init__(self, from_model, to_model):
        self.from_model, self.to_model = from_model, to_model
    
    def predict(self, b, c, castles, passant, halfmoves, fullmoves):
        inp = convert(b, c, castles, passant, halfmoves, fullmoves)
        print(inp)
        board = chess.Board(inp)
        splits = inp.split(' ')
        fen, col = splits[0], splits[1]
        
        if col == 'b': 
            fen = invert_fen(fen)     
            
        arr = fen_to_matrix(fen=fen, piece_dict=debug_piece_dict)
        arr = arr.reshape((1,) + arr.shape)
        from_matrix = self.from_model.predict(arr).reshape((8, 8))
        to_matrix = self.to_model.predict(arr).reshape((8, 8))
        
        if col == 'b': 
            from_matrix = np.flip(from_matrix, axis=0)
            to_matrix = np.flip(to_matrix, axis=0)
            
        moves = []
        for move in list(board.legal_moves):
            from_row, from_col, to_row, to_col = uci_to_row_col(move.uci())
            score = from_matrix[from_row][from_col] * to_matrix[to_row][to_col]
            moves.append((move, score))
            
        moves.sort(key=lambda x: x[1], reverse=True)

        if board.fullmove_number < 4:
            return moves[random.randint(0, max(0, min(len(moves)-1, 5)))][0]
        else:
            return moves[random.randint(0, max(0, min(len(moves)-1, 2)))][0]
        
computer_instance = Computer(from_model, to_model)
            