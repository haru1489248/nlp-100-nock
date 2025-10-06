import random
from typing import Optional

def typoglycemia(text: str, seed: Optional[int] = None) -> str:
    rng = random.Random(seed)  # 再現性が必要なら seed を渡す
    out = []
    for w in text.split():     # スペースで区切る前提
        if len(w) <= 4:
            out.append(w)
            continue # 次の単語に進む
        mid = list(w[1:-1])    # 2文字目から最後の一つ手前まで
        rng.shuffle(mid)       # 中身だけシャッフル
        out.append(w[0] + "".join(mid) + w[-1])
    return " ".join(out)

s = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

print(typoglycemia(s))        # 毎回結果が変わる
print(typoglycemia(s, seed=0))  # こちらは固定結果
