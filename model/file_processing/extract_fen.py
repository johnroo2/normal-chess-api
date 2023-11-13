import sys 
from alive_progress import alive_bar

#python .venv/model/file_processing/extract_fen.py .venv/model/data/evaluations.csv .venv/model/data/fen.txt

count = 0
print("Counting lines...")
with open(sys.argv[1]) as in_file:
  for _ in in_file:
    count += 1
print("Extracting...")      
with open(sys.argv[1]) as in_file:
  with open(sys.argv[2], 'w') as out_file:
    with alive_bar(count) as bar:
      for line in in_file:
        if line == '\n': continue
        ls = line.split('|')[1:]
        #print(ls)
        out_file.write('{}\n'.format(' '.join(ls)))
        bar()
outcount = 0
print("Counting output...")
with open(sys.argv[2]) as out_file:
  for _ in out_file:
    outcount += 1      
print(f"Output: {outcount} lines")       
print("Extract FEN finished")