#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

######## food.csv は、都道府県庁所在地・政令指定都市 52市における
######## 二名以上世帯の年平均外食消費金額（2017～2019年）のデータである。
######## 出典：総務省統計局 家計調査
if __name__ == "__main__":

  ###### データ解析ライブラリ Pandas を用いて CSV ファイルを読み込む。
  ###### food.csv は Excel で読めるよう SHIFT-JIS なので文字コードを指定
  ###### 最初の行が項目名（ジャンル＝列名）であることを指定
  ###### 最初の列も項目名（都市名＝行名）であることを指定
  data = pd.read_csv('food.csv', encoding='shift_jis', header=0, index_col=0)

  ###### 読み込んだデータの各要素にアクセスするには、iat を用いる。
  ###### 例えば 0行目・2列目（札幌市の中華そば）は、次のようにして読み出せる
  #print(data.iat[0, 2])
  ###### 各要素は loc を用い次のように項目名（都市名とジャンル）でも参照できる
  ##print(data.loc['岐阜市', 'すし'])
  
  ###### 読み込んだデータの列名や行名は次のようにして読み出せる。
  ###### (1) 日本そば・うどん、中華そば、すし、和食、中華食、洋食、
  ######     焼肉、ハンバーガー、喫茶代、飲酒代の全 10 ジャンルについて、
  ######     次の平均を求めるコードを参考に、標準偏差および中央値を求めなさい。
  ######     注: 標準偏差は Pandas の関数を用いず教科書の式に従い計算すること
  print("-------平均値-------")
  for col in range(1, len(data.columns)):   # ジャンルそれぞれ繰り返す
    name = data.columns[col]
    sum = 0
    for row in range(len(data.index)):      # 全都市の合計を求める
      sum += data.iat[row, col]
    ave = sum / len(data.index)             # 合計を都市数で割って平均を求める
    print('{0} {1:.0f}'.format(name, ave))  # 小数点以下は四捨五入で表示する
  print("---------標準偏差----------")
  for col in range(1, len(data.columns)):   # ジャンルそれぞれ繰り返す
    sum=0
    ave=0
    sum1=0
    name = data.columns[col]
    for row in range(len(data.index)):      # 全都市の合計を求める
      sum += data.iat[row, col]
    ave = sum / len(data.index) # 合計を都市数で割って平均を求める
    for row1 in range(len(data.index)):      # 全都市の合計を求める
      sum1 += (data.iat[row1, col]-ave)**2
    ave=sum1/len(data.index)
    print('{0} {1:.0f}'.format(name, math.sqrt(ave)))  # 小数点以下は四捨五入で表示する
  print('---------中央値-----------')
  print(data.median())
  print("-------------------------")
###### (2) 全 10 ジャンルの中で、岐阜市が全国一のものをすべて答えなさい。
  print("岐阜市が１位のジャンル")  
  for col in range(1,len(data.columns)):
    name = data.columns[col]
    data1=data.iloc[:,col]
    if data1.idxmax()=="岐阜市":
        print(name)
  
  
  ###### (3) 和食と洋食に対し、ピアソンの積率相関係数を求めなさい。
  ######     また、以下のコードにより和食と洋食の分布を散布図で可視化し、
  ######     概ね直線的な関係があることを確認しなさい。
  japanese = data.loc[:, '和食']   # ジャンルが和食のデータ（列）を全て取り出す
  european = data.loc[:, '洋食']   # 　　〃　　洋食　　〃
  print("-------公式--------")
  avex=0
  avey=0
  Sxx=0
  Syy=0
  Sxy=0
  for col in range(0,len(data.index)):
    avex+=japanese.iat[col]
    avey+=european.iat[col]
  avex/=len(data.index)
  avey/=len(data.index)
  for col in range(0,len(data.index)):
    Sxx+=(japanese.iat[col]-avex)**2
    Syy+=(european.iat[col]-avey)**2
    Sxy+=(japanese.iat[col]-avex)*(european.iat[col]-avey)
  
  r=Sxy/((math.sqrt(Sxx))*(math.sqrt(Syy)))
  print("和食と洋食のピアソン 積率相関係数　公式:{0}".format(r))
  print("------ライブラリ------")
  df=pd.DataFrame({'和食':japanese,'洋食':european})
  df_=df.corr()
  pear=df_.iat[0,1]
  print('和食と洋食のピアソン 積率相関係数: {0}'.format(pear)) 
  plt.scatter(japanese, european) #描画ライブラリ Matplotlib で散布図を描画
  plt.show()
  ###### (4) 和食を x、洋食を y として、単回帰分析を行い、
  ######     単回帰係数、切片および決定係数を求めなさい。
  ######     また求まった決定係数が、(3) で求めた相関係数の二乗と一致することを
  ######     確認しなさい。
  sumx=0
  sumy=0
  avex=0
  avey=0
  Sxx=0
  Syy=0
  Sxy=0
  yi=0
  ei=0
  Se=0
  print("---------公式-------")
  for row in range(len(japanese)):# 全都市の合計を求める
    sumx += japanese[row]
    sumy+=european[row]
    
  avex = sumx / len(japanese)# 合計を都市数で割って平均を求める
  avey=sumy/len(european)
  for row in range(len(japanese)):# 全都市の合計を求める
    Sxx += (japanese[row]-avex)*(japanese[row]-avex)
    Syy+=(european[row]-avey)*(european[row]-avey)
    Sxy+=(european[row]-avey)*(japanese[row]-avex)
  for row in range(len(japanese)):# 全都市の合計を求める
    yi=(Sxy/Sxx)*(japanese[row]-avex)+avey
    ei=yi-european[row]
    Se+=ei*ei 
  print('決定係数:{0}'.format(1-Se/Syy))
  print('単回帰係数:{0}'.format(Sxy/Sxx))
  print('切片:{0}'.format(avey-(Sxy/Sxx)*avex))
  print('y = %.2fx + %.2f' % (Sxy/Sxx,avey-(Sxy/Sxx)*avex ))
  print("----------ライブラリ-------")
  x=df[['和食']]
  y=df[['洋食']]
  model = LinearRegression()
  model.fit(x, y)
  model.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))
  plt.plot(x, y, 'o')
  plt.plot(x, model.predict(x), linestyle="solid")
  plt.show()
  print('y = %.2fx + %.2f' % (model.coef_ , model.intercept_))
  print('単回帰係数:{0}'.format(model.coef_))
  print('切片:{0}'.format(model.intercept_))
  print('決定係数:{0}'.format(model.score(x,y)))
  print('相関係数の２乗:{0}'.format(pear*pear))


# In[ ]:




