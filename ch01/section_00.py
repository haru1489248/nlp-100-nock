from itertools import zip_longest

s1 = "パトカー"
s2 = "タクシー"

result = "".join(a + b for a, b in zip(s1, s2))
print(result)  # パタトクカシーー

s1 = "ABCDEF"
s2 = "xy"

result = "".join((a or "") + (b or "") for a, b in zip_longest(s1, s2))
print(result)  # AxByCDEF
