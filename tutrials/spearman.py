import numpy as np

math    = [6, 4, 5, 10, 2, 8, 3, 9, 1, 7]
english = [10, 1, 4, 9, 3, 8, 6, 5, 2, 7]

def spearman(math, english):
    math = np.array(math)
    english = np.array(english)
    N = len(math)

    return 1 - (6*sum((math -english)**2) / (N*(N**2 - 1)))
    
print(spearman(math ,english)) #0.6727272727272727

# スピアマン相関係数
# 2変数間に、どの程度、順位づけの直線関係があるかを調べる際に使う分析手段がスピアマンの順位相関
