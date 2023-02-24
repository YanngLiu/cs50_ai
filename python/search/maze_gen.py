import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_surfix_number', type=int)
parser.add_argument('-r', '--matrix_row', type=int)
parser.add_argument('-c', '--matrix_col', type=int)
args = parser.parse_args()
fn = args.file_surfix_number
R = args.matrix_row
C = args.matrix_col

lines = []
for _ in range(R):
    line = [' '] * C
    walls = set()
    wantedWalls = int(C * random.randint(20, 40) / 100)
    while len(walls) < wantedWalls:
        i = random.randint(0, C - 1)
        walls.add(i)
    for i in walls:
        line[i] = '#'
    lines.append(line)

sx, sy = random.randint(0, R - 1), random.randint(0, C - 1)
ex, ey = sx, sy
while (ex, ey,) == (sx, sy):
    ex, ey = random.randint(0, R - 1), random.randint(0, C - 1)

lines[sx][sy] = 'A'
lines[ex][ey] = 'B'

with open('maze' + str(fn) + '.txt', 'w') as f:
    for l in lines:
        f.write(''.join(l) + '\n')
    f.write('\n')
