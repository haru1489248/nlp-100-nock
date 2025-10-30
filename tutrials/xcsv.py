import csv

src = 'tutrials/assets/dict.csv'
with open(src, encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        print(line)
