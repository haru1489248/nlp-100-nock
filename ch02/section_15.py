from pathlib import Path
import math

def split_by_lines(src_path: str, N: int):
    src = Path(src_path)
    with src.open(encoding="utf-8") as f:
        total = sum(1 for _ in f)
        per = math.ceil(total / N)

        with src.open(encoding="utf-8") as f:
            for k in range(N):
                out = Path(f"ch02/assets/generates/part_{k:02d}.txt") # 02dは0埋め、dで整数、２で二桁表示
                with out.open("w", encoding="utf-8") as w:
                    for _ in range(per):
                        line = f.readline()
                        if not line:
                            break
                        w.write(line)

split_by_lines("ch02/assets/popular-names.txt", 10)

# splitはmacでは使用できないのでgsplitを使用する。使用できないのはmacがBSD版を使用しているから。GNUの-n l/10書式をサポートしていない
# mac でもGNUnのようなコマンドを使用するためにgsplitを使用する

# gsplit -n l/10 は「できるだけ均等」に分けますが、サイズの差が±1まで許容されます（仕様）

# 以上の理由から先に行を計算する
# wc -l ch02/assets/popular-names.txt

# 計算した結果278行で割り切れるので278を指定してコマンドを打った
# gsplit -d -l 278 --additional-suffix=.txt ch02/assets/popular-names.txt ch02/assets/section_15_generates/cli/part_
