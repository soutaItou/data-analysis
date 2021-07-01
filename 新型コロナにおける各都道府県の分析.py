#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import math
######## corona210113.csv は、①都道府県ごとの推計人口（出展: Wikipedia）、
######## 2021年1月13日現在の新型コロナウィルス感染症における、
######## ②入院者数と③それに対応するベッド数、
######## ④重症者数と⑤それに対応するベッド数、
######## ⑥直近一週間と⑦その前週の陽性者数、（以上、出展: NHK）
######## ⑧感染者数＝療養者数、（出展: Yahoo!）
######## ⑨今年の PCR 検査件数の週平均値、
######## ⑩最新の週あたり感染経路不明の症例数（以上、厚生労働省）
######## のデータである。
######## 出典：
########   Wikipedia
########     https://ja.wikipedia.org/wiki/都道府県の人口一覧
########   NHK 新型コロナ データ一覧 都道府県ごとの感染状況
########     https://www3.nhk.or.jp/news/special/coronavirus/data-widget/
########   Yahoo! 新型コロナウイルス感染症まとめ 都道府県別感染者数
########     https://hazard.yahoo.co.jp/article/20200207
########   厚生労働省  新型コロナウイルス感染症 地域ごとの感染状況等の公表について
########     https://www.mhlw.go.jp/stf/seisakunitsuite/newpage_00016.html
if __name__ == "__main__":

  ###### データ解析ライブラリ Pandas を用いて CSV ファイルを読み込む。
  data = pd.read_csv('corona210113.csv', encoding='shift_jis', header=0, index_col=0)
  ###### (1) 感染状況を表す以下の 7つの指標を計算しなさい。
  ######     1) 入院者の病床使用率　②÷③
  ######     2) 重症者の病床使用率　④÷⑤
  ######     3) 人口十万人あたり療養者数　⑧÷(①÷100,000)
  ######     4) 直近一週間の PCR 検査陽性率　⑥÷⑨
  ######     5) 人口十万人あたり直近一週間の陽性者数　⑥÷(①÷100,000)
  ######     6) 直近一週間とその前週との新規感染者の倍率　⑥÷⑦
  ######     7) 直近一週間の感染経路不明者の割合　⑩÷⑥
  x1 = data.loc[:,'(2)入院者数'] / data.loc[:,'(3)新型コロナ対応ベッド数']
  x2 = data.loc[:,'(4)重症者数'] / data.loc[:,'(5)重症者対応ベッド数']
  x3= data.loc[:,'(8)感染者数'] / (data.loc[:,'(1)推計人口']/100000)
  x4= data.loc[:,'(6)直近一週間の陽性者数'] / data.loc[:,'(9)今年のPCR検査件数（週間あたり）']
  x5= data.loc[:,'(6)直近一週間の陽性者数'] / (data.loc[:,'(1)推計人口']/100000)
  x6= data.loc[:,'(6)直近一週間の陽性者数'] / data.loc[:,'(7)その前週一週間の陽性者数']
  x7= data.loc[:,'(10)感染経路不明症例数'] / data.loc[:,'(6)直近一週間の陽性者数']
  x = np.array([x1, x2, x3, x4, x5, x6, x7]).T.tolist()
  x = pd.DataFrame(x, columns=list('1234567'), index=data.index)
  print("------------------感染状況を表す指標:x------------------")
  print(x)
  #print()
  N, D = x.values.shape
###### (2) 感染者の急増を意味する「ステージ 3」の基準値は次のとおりである。
  ######     1) 20％以上、2) 20％以上、3) 15人以上、4) 10％以上、
  ######     5) 15人以上、6) 1以上、7) 50％以上
  ######     また、感染爆発を意味する「ステージ 4」の基準値は次のとおりである。
  ######     1) 50％以上、2) 50％以上、3) 25人以上、4) 10％以上、
  ######     5) 25人以上、6) 1以上、7) 50％以上
  ######     東海三県それぞれ 7つの指標について、
  ######     「ステージ 1・2」「ステージ 3」「ステージ 4」
  ######     のいずれにあるか求めなさい。
##ステージ３
  a13 = (x1 >= 0.2)
  a23 = (x2 >= 0.2)
  a33 = (x3 >= 15)
  a43 = (x4 >= 0.1)
  a53 = (x5>= 15)
  a63 = (x6 >= 1)
  a73 = (x7 >= 0.5)
  a3 = np.array([a13, a23, a33, a43, a53, a63, a73]).T.tolist()
  a3 = pd.DataFrame(a3, columns=list('1234567'), index=data.index)
  aichi3=a3.loc['愛知県',:]
  gifu3=a3.loc['岐阜県',:]
  mie3=a3.loc['三重県',:]
##ステージ４
  a14 = (x1 >= 0.5)
  a24 = (x2 >= 0.5)
  a34 = (x3 >= 25)
  a44 = (x4 >= 0.1)
  a54 = (x5 >= 25)
  a64 = (x6 >= 1)
  a74 = (x7 >= 0.5)
  a4 = np.array([a14, a24, a34, a44, a54, a64, a74]).T.tolist()
  a4 = pd.DataFrame(a4, columns=list('1234567'), index=data.index)
  aichi4=a4.loc['愛知県',:]
  gifu4=a4.loc['岐阜県',:]
  mie4=a4.loc['三重県',:]
  
  aichi,gifu,mie=np.zeros(D),np.zeros(D),np.zeros(D)
  
  for col in range(0,len(aichi3.index)):
    if aichi4.iat[col]==True:
      aichi[col]=4
    elif aichi3.iat[col]==True:
      aichi[col]=3
    else :
      aichi[col]=2
  for col in range(0,len(gifu3.index)):
    if gifu4.iat[col]==True:
      gifu[col]=4
    elif gifu3.iat[col]==True:
      gifu[col]=3
    else :
      gifu[col]=2
  for col in range(0,len(mie3.index)):
    if mie4.iat[col]==True:
      mie[col]=4
    elif mie3.iat[col]==True:
      mie[col]=3
    else :
      mie[col]=2
  
  name=["入院者の病床使用率","重症者の病床使用率","人口十万人あたり療養者数","直近一週間の PCR 検査陽性率","人口十万人あたり直近一週間の陽性者数","直近一週間とその前週との新規感染者の倍率","直近一週間の感染経路不明者の割合"]
  print("--------------愛知県--------------")
  for col in range(0,7):
    if aichi[col]==4:
      print("愛知県の"+name[col]+":ステージ4")
    elif aichi[col]==3:
      print("愛知県の"+name[col]+":ステージ3")
    else:
      print("愛知県の"+name[col]+":ステージ1.2")
  print("---------------岐阜県---------------")
  for col in range(0,7):
    if gifu[col]==4:
      print("岐阜県の"+name[col]+":ステージ4")
    elif gifu[col]==3:
      print("岐阜県の"+name[col]+":ステージ3")
    else:
      print("岐阜県の"+name[col]+":ステージ1.2")
  print("---------------三重県-----------------")
  for col in range(0,7):
    if mie[col]==4:
      print("三重県の"+name[col]+":ステージ4")
    elif mie[col]==3:
      print("三重県の"+name[col]+":ステージ3")
    else:
      print("三重県の"+name[col]+":ステージ1.2")
  ###### (3) 7つの指標から緊急事態宣言の有無を客観的に評価したい。
  ######     47都道府県のデータ（7指標と実際の緊急事態宣言の有無）を用いて、
  ######     宣言が発令されている 11都府県のグループ G1 と、
  ######     それ以外の 36道県のグループ G0 に対して、
  ######     それぞれマハラノビス距離を計算しなさい。
    
  ###### 都道府県の数 N、指標の数 D

  N, D = x.values.shape
  ###### グループごとのサイズ n0、n1 と 7指標の平均 m0、m1 を計算
  n0, m0 = 0, np.zeros(D)
  n1, m1 = 0, np.zeros(D)
  
  for i in range(N):
    if (data.iloc[i, 0] == 'なし'):
      n0+=1
      for n in range(D):
        m0[n]+= x.iloc[i,n]
      continue
    else:
      n1+=1
      for n in range(D):
        m1[n] += x.iloc[i,n]
      continue
  for n in range(D):
    m0[n]/=n0
    m1[n]/=n1
  print("----------m0-----------------")
  print(m0)
  print("----------m1-----------------")
  print(m1)
  ###### グループごとの 7指標の分散共分散行列 S0, S1 を計算
  ##N47 D 7
  tmp1=0
  tmp2=0
  S0, S1 = np.zeros((D, D)), np.zeros((D, D))
  print("---------------S0--------------")
  for i in range(D):
    for j in range(D):
      for k in range(N):  
        if (data.iloc[k, 0] == 'なし'):
          tmp1+=(x.iloc[k,i]-m0[i])*(x.iloc[k,j]-m0[j])
        elif (data.iloc[k, 0] == 'あり'):
          tmp2+=(x.iloc[k,i]-m1[i])*(x.iloc[k,j]-m1[j])
      tmp1/=n0
      tmp2/=n1
      S0[i][j]=tmp1
      S1[i][j]=tmp2
      tmp1=0
      tmp2=0
  #print(S0)
  print("---------------S1---------------")
   
  #print(S1)
  ###### 各グループに対するマハラノビス距離 g0、g1 とその差 gd を計算
  g0, g1, gd = [], [], []
  s0_=np.linalg.inv(S0)
  s1_=np.linalg.inv(S1)
  
  
  for i in range(N):
    t = x.values[i,:]
    l0=np.array([t-m0])
    l1=np.array([t-m1])
    d0 =math.sqrt(np.dot(np.dot(l0,s0_),l0.T))
    d1 =math.sqrt(np.dot(np.dot(l1,s1_),l1.T))
    
    g0.append(d0)
    g1.append(d1)
    gd.append(d0 - d1)
  print(gd)

  for i in range(N):
    if(gd[i]>=0):
      print(data.index[i]+":グループ１")
    else:
      print(data.index[i]+":グループ0")  
  result = pd.DataFrame(np.array([g0, g1, gd]).T, columns=list('01D'), index=data.index)
  print(result)



# In[ ]:





# In[ ]:




