from collections import deque # 両端に余白があるから両端の追加・削除が高速（O(1)）（Double-Ended QUEue）
from pathlib import Path

DATA = Path("ch02/assets/popular-names.txt")

with DATA.open(encoding="utf-8") as f:
    tail = deque(f, maxlen=10)

for line in tail:
    print(line, end="")

with open('ch02/assets/popular-names.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[-10:]:
        print(line)

# tailコマンド
# tail -n 10 ch02/assets/popular-names.txt
