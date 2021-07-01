#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

######## asia1.csv は、東アジア・東南アジア各国・地域の産業構造割合を表している。
######## （注:「その他」を抜いたため合計は 100 にならない、またデータ不足のため
########   一部の国・地域を除外している）
if __name__ == "__main__":

  ###### データ解析ライブラリ Pandas を用いて CSV ファイルを読み込む。
  data = pd.read_csv('asia1.csv', encoding='shift_jis', header=0, index_col=0)
  
  ###### (4) 主成分分析を行い、各国・地域を 2次元の新たな変量 x, y で表し、
  ######     XY 平面にプロットして結果例のようになることを確認しなさい。
  ######     また、そのときの累積寄与率を求めなさい。

  ###### 国・地域の数 N、項目の数 D
  N, D = data.values.shape

  ###### 平均 m を計算
  m = np.zeros(D)
  for i in range(D):
    m[i] = np.mean(data.values[:, i])
  print("------------平均mーーーーーーーーーーーーーー")
  print(m)
  ###### 分散共分散行列 S を計算
  S = np.zeros((D, D))
  #： 
  tmp=0
  for i in range(D):
    for j in range(D):
      for k in range(N):  
          tmp+=(data.iloc[k,i]-m[i])*(data.iloc[k,j]-m[j])
      tmp/=N
      S[i][j]=tmp
      tmp=0
  print("----------分散共分散行列S------------")
  print(S)
      

  ###### 固有値分解して第一・第二主成分のベクトル v1、v2 を取得
  eval, evec = np.linalg.eig(S)
  v1, v2 = evec[0], evec[1]
  

  ###### 二次元データに要約
  x, y = [], []
  for i in range(N):
    t=data.iloc[i,:]
    l=np.array([t-m])
    x.append(np.dot(l,v1))
    y.append(np.dot(l,v2))
    continue
  
  ###### 日本語フォントの読み込み（Windows 用）
  #fp = FontProperties(fname=r'C:\WINDOWS\Fonts\msgothic.ttc', size=10)
  plt.rcParams['font.family'] = 'AppleGothic'
  ###### XY 平面にプロット
  plt.scatter(x, y)
  for i, l in enumerate(data.index):
    plt.annotate(l, (x[i], y[i]))
  plt.show()

  ###### 累積寄与率を計算・表示
  i=0
  for l in range(D):
    i+=eval[l]
    kiyoritu= i/np.sum(eval)
    print(str(l+1)+"の累積寄与率")
    print(":")
    print(kiyoritu)
#   l2=(eval[0]+eval[1])/(eval[0]+eval[1]+eval[2]+eval[3]+eval[4]+eval[5])
#   print("第２主成分までの累積寄与率:")
#   print(l2)


# In[ ]:





# In[ ]:




