import time
import sys
### Changes you make to this file will not persist to our testing servers

RESULT_LOCATION = 'results/output.txt'
DATASET_LOCATION = 'data.csv'
SCORE_LOCATION = 'results/score.txt'

argc = len(sys.argv)

if (argc > 1):
    RESULT_LOCATION = sys.argv[1]
if (argc > 2): 
    DATASET_LOCATION = sys.argv[2]
if (argc > 3):
    SCORE_LOCATION = sys.argv[3]

err2_tally = 0
y2_tally = 0

with open(DATASET_LOCATION, 'r') as dp:
    with open(RESULT_LOCATION, 'r') as sp:
        i = 0
        while(True):
            line = dp.readline()
            if not line:
                break
            if i == 0:
                i += 1
                continue # don't read first line of data because it contains headers
            y_true = float(line.split(',')[-1][:-1])

            guess_line = sp.readline()[:-1] 
            y_guess = float(guess_line)
            
            err2_tally += (y_true - y_guess) ** 2
            y2_tally += y_true ** 2

r2 = 1 - err2_tally / y2_tally

print(f"You achieved an r2 value of {r2}.")

with open(SCORE_LOCATION, 'w') as wp:
    wp.write(str(r2))
