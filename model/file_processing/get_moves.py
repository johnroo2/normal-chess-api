import numpy as np
from alive_progress import alive_bar
from preprocess import piece_dict
import sys 

#python .venv/model/file_processing/get_moves.py .venv/model/data/fen.txt .venv/model/data/moves.txt

def fen_to_matrix(fen):
  row_arr = []
  rows = fen.split("/")
  for row in rows:
    arr = []
    for ch in str(row):
      if ch.isdigit():
        for _ in range(int(ch)):
          arr.append(piece_dict['.'])
      else:
        arr.append(piece_dict[ch])
    row_arr.append(arr)
  return np.array(row_arr)

previous = None
previous_fen = None
count = 0

print("Counting lines...")
with open(sys.argv[1]) as in_file:
  for _ in in_file:
      count += 1

print("Retrieving moves...")
with open(sys.argv[1]) as in_file:
  with open(sys.argv[2], 'w') as out_file:
    with alive_bar(count) as bar:
      for line in in_file:
        sline = line.rstrip()
        if sline == '\n' or sline == '' or sline.startswith('fen'): 
          bar()
          continue
        fen = sline.split(" ")[0]
        move = sline.split(" ")[1]
        current = np.argmax(fen_to_matrix(fen), axis=-1)

        if previous is not None:
          out_file.write('{} {}  {}  {}\n'.format(previous_fen, move, 
            np.argmin(current - previous), np.argmax(current - previous)))

        previous = current
        previous_fen = fen
        bar()
      
print("Get moves finished")