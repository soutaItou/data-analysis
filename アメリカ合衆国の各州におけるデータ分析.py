#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
######## election.csv は、アメリカ合衆国のアラスカ・ハワイを除く 48州について、
######## 人口、面積、人口密度、年収中央値、
######## タイムゾーン（東部 ET、中部 CT、山岳 MT、太平洋 PT）{※1}、
######## 家庭で英語を使う世帯の割合、人種の割合{※2}、および、
######## 2020年大統領選挙における二大政党得票数の割合{※3}を示したデータである。
######## 出典：Wikipedia ほか
######## ※1 州によっては 2つのタイムゾーンを有する場合がある
######## ※2 その他を除外しており合計は 100% にはならない
######## ※3 二大政党以外の得票は考慮していない、2020年11月28日現在
if __name__ == "__main__":

  ###### データ解析ライブラリ Pandas を用いて CSV ファイルを読み込む。
  data = pd.read_csv('election.csv', encoding='shift_jis', header=0, index_col=0)
  print("-------------5------------")
  ###### (5) タイムゾーンと共和党得票率との相関比を求めなさい。
  ######    （複数のタイムゾーンを有する州はいずれのグループにも属すると考え、
  ######      例えばフロリダ州の得票率"51.7%"は、ET と CT のグループ両方に加える）
  ######     なお参考までに、以下に級内変動 sw を求めるコードを示す。
  timezones = { 'ET': [], 'CT': [], 'MT': [], 'PT': []}
  for i,row in data.iterrows():
    for tz in row['タイムゾーン'].split(' '):
      timezones[tz].append(row['共和党'])
  sw = 0
  sb=0
  ave=0
  S=0
  num=0
  ave1=0
  for tz in timezones.keys():
    ave1=0
    m = 0
    for x in timezones[tz]: 
        m += x
    ave1=m/len(timezones[tz])
    for x in timezones[tz]:  
        sw += (x-ave1) * (x-ave1)
        num+=1
    print('{0} {1}'.format(tz, ave1))
    print(len(timezones[tz]))
    print(m)
    ave+=m
  ave/=num
  print(ave)
  m=0
  for tz in timezones.keys():
    m=0
    for x in timezones[tz]:  m += x
    m /= len(timezones[tz])
    for x in timezones[tz]:  sb+=(m-ave)**2
  for tz in timezones.keys():
    m = 0
    for x in timezones[tz]:  S += (x-ave)**2
  print("sw")
  print(sw)
  print("sb")
  print(sb)
  print("S")
  print(S)
  print("求める相関比:{0}".format(1-sw/S))
  
  print("-------------6----------------")
  ###### (6) (5) のダミー変数も含めた全項目の中で、民主党得票率と相関が強いのは
  ######     順に年収中央値、アジア系人口比率、英語世帯割合なのを確認しなさい。
  ######     次に、年収中央値を x、英語世帯割合を y、民主党得票率を r として、
  ######     重回帰分析 r=ax+by+c の偏回帰係数 a・b と切片 c、および、
  ######     自由度調整済み決定係数を求めなさい。
  ######    （python のライブラリを用いてもよいが、教科書の公式を用いた場合と
  ######     検算を行うこと）
  r=data.loc[:, "民主党"]
  for i in range(0, len(data.columns)): 
    Sxx=0
    Srr=0
    Sxr=0
    avex=0
    aver=0
    name= data.columns[i]
    if name=="タイムゾーン":
      continue
    #if name=="民主党":
      #continue
    r=data.loc[:, "民主党"]
    x=data.loc[:, name]
    for col in range(0,len(data.index)):
      avex+=x.iat[col]
      aver+=r.iat[col]
    avex/=len(data.index)
    aver/=len(data.index)
    for col in range(0,len(data.index)):
     Sxx+=(x.iat[col]-avex)**2
     Srr+=(r.iat[col]-aver)**2
     Sxr+=(x.iat[col]-avex)*(r.iat[col]-aver)
    r=Sxr/(math.sqrt(Srr)*math.sqrt(Sxx))
    print("民主党得票率と"+name+" 積率相関係数　公式:{0}".format(r))
  print("よって相関が強い順に年収中央値,アジア系人口比率、英語世帯割合となる")
  x=data.loc[:, "年収中央値[USD]"]
  y=data.loc[:, "英語話者"]
  r=data.loc[:,"民主党"]
  Sxx=0
  Syy=0
  Srr=0
  Sxy=0
  Syr=0
  Sxr=0
  avex=0
  avey=0
  aver=0
  ri=0
  Se=0
  n=len(data.index)
  p=2
  R2=0
  for col in range(0,len(data.index)):
    avex+=x.iat[col]
    avey+=y.iat[col]
    aver+=r.iat[col]
  avex/=len(data.index) 
  avey/=len(data.index) 
  aver/=len(data.index)
  for col in range(0,len(data.index)):
     Sxx+=(x.iat[col]-avex)**2
     Syy+=(y.iat[col]-avey)**2
     Srr+=(r.iat[col]-aver)**2   
     Sxy+=(x.iat[col]-avex)*(y.iat[col]-avey)
     Syr+=(y.iat[col]-avey)*(r.iat[col]-aver)
     Sxr+=(x.iat[col]-avex)*(r.iat[col]-aver)
  l1=[[Sxx,Sxy],[Sxy,Syy]]
  l2=[[Sxr],[Syr]]
  l1=np.matrix(l1)
  l2=np.matrix(l2)
  _l1=l1**-1
  a_b=_l1*l2
  a=a_b[0][0]
  b=a_b[1][0]
  c=aver-a*avex-b*avey
  for col in range(0,len(data.index)):
    ri=a*x.iat[col]+b*y.iat[col]+c
    ei=ri-r.iat[col]
    Se+=ei**2
  print(Se)
  R2=1-(Se*(n-1))/(Srr*(n-p-1))
  
  print("偏回帰係数a:{0}".format(a))
  print("偏回帰係数b:{0}".format(b))
  print("切片c:{0}".format(c))
  print("r={0}x+{1}y+{2}".format(a,b,c))
  print( "自由度調整済み決定係数R2:{0}".format(R2))
  

  print("-----------7-----------")
  ###### (7) (6) の重回帰分析について、人口比率のいずれかひとつを z として、
  ######     重回帰分析 r=ax+by+cz+d とすることで精度（決定係数）を改善したい。
  ######     多重共線性にも注意し、どれを追加するのが最もよいか、答えなさい。
  max=0
  for i in range(0, len(data.columns)): 
    avex=0
    avey=0
    aver=0
    avez=0
    Sxx=0
    Syy=0
    Szz=0
    Srr=0
    Sxy=0
    Sxz=0
    Sxr=0
    Syz=0
    Syr=0
    Szr=0
    Se=0
    ei=0
    ri=0
    p=3
    n=len(data.index)
    name= data.columns[i]
    if name!="白人"and name!="黒人"and name!="ヒスパニック"and name!="アジア系"and name!="ネイティブ":
     continue
    z=data.loc[:, name]
    for col in range(0,len(data.index)):
     avex+=x.iat[col]
     avey+=y.iat[col]
     avez+=z.iat[col]
     aver+=r.iat[col]
    avex/=len(data.index) 
    avey/=len(data.index) 
    avez/=len(data.index)  
    aver/=len(data.index)
    for col in range(0,len(data.index)):
      Sxx+=(x.iat[col]-avex)**2
      Syy+=(y.iat[col]-avey)**2
      Szz+=(z.iat[col]-avez)**2
      Srr+=(r.iat[col]-aver)**2   
      Sxy+=(x.iat[col]-avex)*(y.iat[col]-avey)
      Sxz+=(x.iat[col]-avex)*(z.iat[col]-avez)   
      Sxr+=(x.iat[col]-avex)*(r.iat[col]-aver)
      Syz+=(y.iat[col]-avey)*(z.iat[col]-avez)
      Syr+=(y.iat[col]-avey)*(r.iat[col]-aver)
      Szr+=(z.iat[col]-avez)*(r.iat[col]-aver)
    l1=[[Sxx,Sxy,Sxz],[Sxy,Syy,Syz],[Sxz,Syz,Szz]]
    l2=[[Sxr],[Syr],[Szr]]
    l1=np.matrix(l1)
    l2=np.matrix(l2)
    _l1=l1**-1
    a_b_c=_l1*l2
    a=a_b_c[0][0]
    b=a_b_c[1][0]
    c=a_b_c[2][0]
    d=aver-a*avex-b*avey-c*avez
    print("z="+name+"とした時の重回帰分析:")
    print("r={0}x+{1}y+{2}z+{3}".format(a,b,c,d))
    for col in range(0,len(data.index)):
     ri=a*x.iat[col]+b*y.iat[col]+c*z.iat[col]+d
     ei=ri-r.iat[col]
     Se+=ei**2
    R2=1-(Se*(n-1))/(Srr*(n-p-1)) 
    rxr=Sxr/(math.sqrt(Srr)*math.sqrt(Sxx))
    ryr=Syr/(math.sqrt(Srr)*math.sqrt(Syy))
    rzr=Szr/(math.sqrt(Srr)*math.sqrt(Szz))
    print("自由度調整済み決定係数R2:{0}".format(R2))
    print("それぞれの相関係数　xとr:{0} yとr:{1} zとr:{2}".format(rxr,ryr,rzr))
    if (np.sign(rxr)==np.sign(a)) and (np.sign(ryr)==np.sign(b)) and (np.sign(rzr)==np.sign(c)):
     print("多重共線性の問題がない")
     if max<=R2:
        name1=name    
        max=R2
     print("現在最大の自由度調整済み決定係数R2:{0}".format(max))
     print("z:"+name)
    else: 
     print("多重共線性の問題がある")


  print("zは"+name1+"にすべきである")


# In[ ]:





# In[ ]:




