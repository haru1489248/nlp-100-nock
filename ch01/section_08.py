def cipher(s: str) -> str:
    # 英小文字 a(97) ~ z(122) だけを 219 - ord(c) に変換
    return "".join(
        chr(219 - ord(c)) if 'a' <= c <= 'z' else c
        for c in s
    )

msg = "Hello, World! 123 abc xyz."
ans_1 = cipher(msg)
print(ans_1)
ans_2 = cipher(ans_1) # 2回暗号化すると元に戻る
print(ans_2)
