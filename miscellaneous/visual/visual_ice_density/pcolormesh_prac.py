import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0,5,1)
b = np.arange(0,-5,-1)
z = np.arange(0,25,1)
zz= z.reshape((5,5))
aa,bb = np.meshgrid(a,b)
plt.pcolormesh(aa,bb,zz)
#plt.pcolormesh(a,b,z)
plt.axis("image")
plt.colorbar()
plt.show()
grids = np.vstack([aa.ravel(), bb.ravel()])
z = z.ravel()

#pcolormeshの描画の順番を確認

"""
900*900 -> 145*145 のアルゴリズム
密接度の方：
x.linspaceとyのやつを取得
右,下に行くほど大きくなるように設定
一回lat,lon,密接度の形に直す？

風，氷の速度の方：
軸を同じように設定
各グリッドに対して，密接度のどの正方形にそのグリッドがあるかを調べて，その値を密接度（145*145）の新たな配列に代入
これを日数分繰り返す
"""