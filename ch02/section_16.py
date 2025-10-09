import random
from pathlib import Path

def shuffle_lines(src, dst):
    rng = random.Random() # 乱数生成器インスタンスを作成している（seedを指定しない場合はOSのエントロピーで初期化され毎回違う乱数になる）（seedを指定する場合は乱数が同じになる）
    src = Path(src)
    dst = Path(dst)
    with src.open(encoding="utf-8") as f:
        lines = f.readlines()
        rng.shuffle(lines)
    with dst.open("w", encoding="utf-8") as w:
        w.writelines(lines)

shuffle_lines("ch02/assets/popular-names.txt", "ch02/assets/section_16_generates/shuffled.txt")

# shufコマンド
# gshuf ch02/assets/popular-names.txt > ch02/assets/section_16_generates/cli/shuffled.txt
