import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', nargs='*')
args = parser.parse_args()

for filename in args.files:
    with open(filename) as f:
        contents = f.read()

    cells =  contents.count(" ")
    walls =  contents.count("#")
    print(cells, walls, str(int(walls/len(contents)*100))+'%')