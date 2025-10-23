import numpy as np

# arrangeで15まで、reshapeで３*5の行列を作成する
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape) # return (3, 5) 3かけ5の行列
print(a.ndim) # 次元数を返す 今回は 2
print(a.dtype.name) # 配列の中の型
print(a.itemsize) # 配列の各要素のバイト数
print(type(a)) # class ndarray


x = np.array(1)            # OK: 0次元配列（スカラー配列）; shape: ()
y = np.array([1])          # OK: 1次元配列（長さ1）; shape: (1,)
z = np.array((1, 2, 3))    # OK: 1次元配列; shape: (3,)
n = np.array([[1, 2], [3, 4]])  # OK: 2次元配列; shape: (2, 2)
print(n)
s = np.array([(1, 2, 3), (4, 5, 6)])
print(s)
# NG = np.array(1, 2, 3, 4)   # NG: 引数を複数並べたので TypeError

complex = np.array([[1,2], [3,4]], dtype=complex) # dtypeを指定できる complex=複素数
print(complex) # 整数なので虚数部は0

zeros = np.zeros([3, 4], dtype=np.int16) # np.int16は固定長の16bit符号付きの整数型 範囲(-32768 ～ 32767)
print(zeros)
ones = np.ones([2, 3, 4], dtype=np.int16)
print(ones)
empty = np.empty([2, 3]) # 初期化しない行列を作成する、使う前に代入する必要がある
print(empty)
