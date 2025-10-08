from pathlib import Path
from itertools import islice

Data = Path('ch02/assets/popular-names.txt')
with Data.open(encoding='utf-8') as f:
    lines = islice(f, 10)
    for line in lines:
        print(line.replace('\t', ' '), end='')


# trコマンド
# head -n 10 ch02/assets/popular-names.txt | tr '\t' ' '

# sedコマンド
# sは置換コマンド、gは行の中全てという意味。ない場合は行の中で初めに見つかったものを置換する
# MacOS系（BSD sed)
# head -n 10 ch02/assets/popular-names.txt | sed $'s/\t/ /g'
# GNU sedならこれでも可
# head -n 10 ch02/assets/popular-names.txt | sed 's/\t/ /g'

# expandコマンド
# head -n 10 ch02/assets/popular-names.txt | expand -t 1

