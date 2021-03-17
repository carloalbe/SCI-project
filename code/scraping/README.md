# Google Patents Scraping


```python
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
import urllib.request
import os
from google_patent_scraper import scraper_class
import json
import re
import timeit
```


```python
ids = pd.read_csv(r"..\..\data\gp-q2.csv",skipinitialspace=True)
ids.isnull().sum()

```




    id                                0
    title                             0
    assignee                         15
    inventor/author                 383
    priority date                   324
    filing/creation date              6
    publication date                  1
    grant date                    13623
    result link                       0
    representative figure link    10002
    dtype: int64




```python
codes = [p[p.find('patent/')+7:-3] for p in ids['result link']]
ids['code'] = codes
ids.set_index(ids.code,inplace=True)
ids['citations'] = [[] for i in ids.index]
ids 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>assignee</th>
      <th>inventor/author</th>
      <th>priority date</th>
      <th>filing/creation date</th>
      <th>publication date</th>
      <th>grant date</th>
      <th>result link</th>
      <th>representative figure link</th>
      <th>code</th>
      <th>citations</th>
    </tr>
    <tr>
      <th>code</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>KR100866792B1</td>
      <td>KR-100866792-B1</td>
      <td>Method and apparatus for generating face descr...</td>
      <td>삼성전자주식회사</td>
      <td>문영수, 박규태, 자오지아리, 황산성, 황원준</td>
      <td>2007-01-10</td>
      <td>2007-01-10</td>
      <td>2008-11-04</td>
      <td>2008-11-04</td>
      <td>https://patents.google.com/patent/KR100866792B...</td>
      <td>https://patentimages.storage.googleapis.com/b5...</td>
      <td>KR100866792B1</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>KR100703693B1</td>
      <td>KR-100703693-B1</td>
      <td>System and method for face recognition</td>
      <td>삼성전자주식회사</td>
      <td>김정배, 이종하</td>
      <td>2005-01-13</td>
      <td>2005-01-13</td>
      <td>2007-04-05</td>
      <td>2007-04-05</td>
      <td>https://patents.google.com/patent/KR100703693B...</td>
      <td>https://patentimages.storage.googleapis.com/5a...</td>
      <td>KR100703693B1</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>EP1732028B1</td>
      <td>EP-1732028-B1</td>
      <td>System and method for detecting an eye</td>
      <td>Delphi Technologies, Inc.</td>
      <td>Riad I. Hammoud, Andrew L. Wilhelm</td>
      <td>2005-06-10</td>
      <td>2006-05-31</td>
      <td>2012-01-11</td>
      <td>2012-01-11</td>
      <td>https://patents.google.com/patent/EP1732028B1/en</td>
      <td>https://patentimages.storage.googleapis.com/88...</td>
      <td>EP1732028B1</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>US9396319B2</td>
      <td>US-9396319-B2</td>
      <td>Method of criminal profiling and person identi...</td>
      <td>Laird H. Shuart, Marcia L Shuart, Sharon E Jan...</td>
      <td>Laird H. Shuart, Marcia L Shuart, Sharon E Jan...</td>
      <td>2013-09-30</td>
      <td>2014-09-24</td>
      <td>2016-07-19</td>
      <td>2016-07-19</td>
      <td>https://patents.google.com/patent/US9396319B2/en</td>
      <td>https://patentimages.storage.googleapis.com/ff...</td>
      <td>US9396319B2</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>KR100846500B1</td>
      <td>KR-100846500-B1</td>
      <td>Method and apparatus for recognizing face usin...</td>
      <td>삼성전자주식회사</td>
      <td>기석철, 문영수, 박규태, 이종하, 황산성, 황원준</td>
      <td>2006-11-08</td>
      <td>2006-11-08</td>
      <td>2008-07-17</td>
      <td>2008-07-17</td>
      <td>https://patents.google.com/patent/KR100846500B...</td>
      <td>https://patentimages.storage.googleapis.com/c9...</td>
      <td>KR100846500B1</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>WO2007077263A1</td>
      <td>WO-2007077263-A1</td>
      <td>Method for detecting chemical species and devi...</td>
      <td>Ramem, S.A.</td>
      <td>Emilio Ramiro Arcas, Angel RIVERO JIMÉNEZ</td>
      <td>2005-12-30</td>
      <td>2005-12-30</td>
      <td>2007-07-12</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/WO2007077263...</td>
      <td>NaN</td>
      <td>WO2007077263A1</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>JPH05233978A</td>
      <td>JP-H05233978-A</td>
      <td>Fire detection system</td>
      <td>Kokusai Gijutsu Kaihatsu Kk, 国際技術開発株式会社</td>
      <td>Masanori Hirasawa, 正憲 平澤</td>
      <td>1992-02-25</td>
      <td>1992-02-25</td>
      <td>1993-09-10</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/JPH05233978A/en</td>
      <td>NaN</td>
      <td>JPH05233978A</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>CN112307080A</td>
      <td>CN-112307080-A</td>
      <td>Power grid power supply loss analysis method d...</td>
      <td>国网福建省电力有限公司莆田供电公司, 国网福建省电力有限公司, 厦门亿力吉奥信息科技有限公司</td>
      <td>陈晶腾, 魏海斌, 吴敏辉, 林培聪, 周天华, 陈辉河, 蒋雷震, 林宇澄, 陈芳, 林立...</td>
      <td>NaN</td>
      <td>2020-10-15</td>
      <td>2021-02-02</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/CN112307080A/en</td>
      <td>NaN</td>
      <td>CN112307080A</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>UA95274U</td>
      <td>UA-95274-U</td>
      <td>Automated system of electronic marking and ins...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-10-15</td>
      <td>2014-10-15</td>
      <td>2014-12-10</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/UA95274U/en</td>
      <td>NaN</td>
      <td>UA95274U</td>
      <td>[]</td>
    </tr>
    <tr>
      <td>TW200513303A</td>
      <td>TW-200513303-A</td>
      <td>Campaign doll</td>
      <td>Teamwell Technology Corp</td>
      <td>jian-zhong Cai</td>
      <td>2003-10-06</td>
      <td>2003-10-06</td>
      <td>2005-04-16</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/TW200513303A/en</td>
      <td>NaN</td>
      <td>TW200513303A</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
<p>22573 rows × 12 columns</p>
</div>




```python
#start = timeit.default_timer()  
for i in ids.index: 
    
    #request url
    fp = urllib.request.urlopen("https://patents.google.com/patent/"+i+"/en")
    #print("https://patents.google.com/patent/"+i+"/en")
    soup = BeautifulSoup(fp.read(), 'html')
    fp.close()
    
    #Get abstract
    abstract = soup.find('abstract')
    if abstract != None: ids.loc[i,'abstract'] = abstract.text
   
    #Get connections to other patents on google patents
    links = []
    trs = soup.find_all('tr',itemprop='similarDocuments')
    if len(trs)>=0:
        for tr in trs:
            l=tr.find('a',href=True)
            if l['href'] != '/patent/US20190146965A1/en' and l['href'].find('/patent/') !=-1 and l['href'].find('/en') !=-1 :
                links.append(l['href'][l['href'].find('/patent/')+8:-3])
    ids.loc[i,'citations'] = list(set(links))
    
    #Get code
    codes = soup.find_all('span',itemprop='Code')
    if len(codes)>=3:
        ids.loc[i,'class'] = codes[3].get_text()[:6]
    
    
    
   
#stop = timeit.default_timer()    
#print('Time: ', stop - start) 
```


```python
ids
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>assignee</th>
      <th>inventor/author</th>
      <th>priority date</th>
      <th>filing/creation date</th>
      <th>publication date</th>
      <th>grant date</th>
      <th>result link</th>
      <th>representative figure link</th>
      <th>code.1</th>
      <th>citations</th>
      <th>abstract</th>
      <th>class</th>
    </tr>
    <tr>
      <th>code</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>KR100866792B1</td>
      <td>KR-100866792-B1</td>
      <td>Method and apparatus for generating face descr...</td>
      <td>삼성전자주식회사</td>
      <td>문영수, 박규태, 자오지아리, 황산성, 황원준</td>
      <td>2007-01-10</td>
      <td>2007-01-10</td>
      <td>2008-11-04</td>
      <td>2008-11-04</td>
      <td>https://patents.google.com/patent/KR100866792B...</td>
      <td>https://patentimages.storage.googleapis.com/b5...</td>
      <td>KR100866792B1</td>
      <td>['US20150178547A1', 'KR100707195B1', 'EP171651...</td>
      <td>\n\n 본 발명은 확장 국부 이진 패턴(Extended Local Binary P...</td>
      <td>G06K9/</td>
    </tr>
    <tr>
      <td>KR100703693B1</td>
      <td>KR-100703693-B1</td>
      <td>System and method for face recognition</td>
      <td>삼성전자주식회사</td>
      <td>김정배, 이종하</td>
      <td>2005-01-13</td>
      <td>2005-01-13</td>
      <td>2007-04-05</td>
      <td>2007-04-05</td>
      <td>https://patents.google.com/patent/KR100703693B...</td>
      <td>https://patentimages.storage.googleapis.com/5a...</td>
      <td>KR100703693B1</td>
      <td>['US10728242B2', 'US8433922B2', 'US9122913B2',...</td>
      <td>\n\n 본 발명은 얼굴 인식에 관한 것으로서, 본 발명의 실시에 따른 얼굴 인식 ...</td>
      <td>G06K9/</td>
    </tr>
    <tr>
      <td>EP1732028B1</td>
      <td>EP-1732028-B1</td>
      <td>System and method for detecting an eye</td>
      <td>Delphi Technologies, Inc.</td>
      <td>Riad I. Hammoud, Andrew L. Wilhelm</td>
      <td>2005-06-10</td>
      <td>2006-05-31</td>
      <td>2012-01-11</td>
      <td>2012-01-11</td>
      <td>https://patents.google.com/patent/EP1732028B1/en</td>
      <td>https://patentimages.storage.googleapis.com/88...</td>
      <td>EP1732028B1</td>
      <td>['EP0989517B1', 'US7206435B2', 'US6028949A', '...</td>
      <td>NaN</td>
      <td>G06K9/</td>
    </tr>
    <tr>
      <td>US9396319B2</td>
      <td>US-9396319-B2</td>
      <td>Method of criminal profiling and person identi...</td>
      <td>Laird H. Shuart, Marcia L Shuart, Sharon E Jan...</td>
      <td>Laird H. Shuart, Marcia L Shuart, Sharon E Jan...</td>
      <td>2013-09-30</td>
      <td>2014-09-24</td>
      <td>2016-07-19</td>
      <td>2016-07-19</td>
      <td>https://patents.google.com/patent/US9396319B2/en</td>
      <td>https://patentimages.storage.googleapis.com/ff...</td>
      <td>US9396319B2</td>
      <td>[]</td>
      <td>\nA method of criminal profiling and person id...</td>
      <td>G06F21</td>
    </tr>
    <tr>
      <td>KR100846500B1</td>
      <td>KR-100846500-B1</td>
      <td>Method and apparatus for recognizing face usin...</td>
      <td>삼성전자주식회사</td>
      <td>기석철, 문영수, 박규태, 이종하, 황산성, 황원준</td>
      <td>2006-11-08</td>
      <td>2006-11-08</td>
      <td>2008-07-17</td>
      <td>2008-07-17</td>
      <td>https://patents.google.com/patent/KR100846500B...</td>
      <td>https://patentimages.storage.googleapis.com/c9...</td>
      <td>KR100846500B1</td>
      <td>['US10565433B2', 'US20110135166A1']</td>
      <td>\n\n 본 발명은 확장된 가보 웨이브렛 특징 들(Gabor Wavelet Feat...</td>
      <td>G06K9/</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>WO2007077263A1</td>
      <td>WO-2007077263-A1</td>
      <td>Method for detecting chemical species and devi...</td>
      <td>Ramem, S.A.</td>
      <td>Emilio Ramiro Arcas, Angel RIVERO JIMÉNEZ</td>
      <td>2005-12-30</td>
      <td>2005-12-30</td>
      <td>2007-07-12</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/WO2007077263...</td>
      <td>NaN</td>
      <td>WO2007077263A1</td>
      <td>['US6495823B1', 'US10643834B2', 'US6847446B2',...</td>
      <td>\r\nThe invention relates to a method for dete...</td>
      <td>G01N27</td>
    </tr>
    <tr>
      <td>JPH05233978A</td>
      <td>JP-H05233978-A</td>
      <td>Fire detection system</td>
      <td>Kokusai Gijutsu Kaihatsu Kk, 国際技術開発株式会社</td>
      <td>Masanori Hirasawa, 正憲 平澤</td>
      <td>1992-02-25</td>
      <td>1992-02-25</td>
      <td>1993-09-10</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/JPH05233978A/en</td>
      <td>NaN</td>
      <td>JPH05233978A</td>
      <td>['US4692750A', 'US4759069A', 'US20150166010A1'...</td>
      <td>\r\nPURPOSE:To remove various kinds of noise s...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>CN112307080A</td>
      <td>CN-112307080-A</td>
      <td>Power grid power supply loss analysis method d...</td>
      <td>国网福建省电力有限公司莆田供电公司, 国网福建省电力有限公司, 厦门亿力吉奥信息科技有限公司</td>
      <td>陈晶腾, 魏海斌, 吴敏辉, 林培聪, 周天华, 陈辉河, 蒋雷震, 林宇澄, 陈芳, 林立...</td>
      <td>NaN</td>
      <td>2020-10-15</td>
      <td>2021-02-02</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/CN112307080A/en</td>
      <td>NaN</td>
      <td>CN112307080A</td>
      <td>['CN101179195B', 'CN103605757B', 'CN106502772A...</td>
      <td>\r\nThe invention relates to a power grid powe...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>UA95274U</td>
      <td>UA-95274-U</td>
      <td>Automated system of electronic marking and ins...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-10-15</td>
      <td>2014-10-15</td>
      <td>2014-12-10</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/UA95274U/en</td>
      <td>NaN</td>
      <td>UA95274U</td>
      <td>['EP2912547B1', 'BR112018016212A2', 'US1006806...</td>
      <td>Автоматизована система електронного маркува...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>TW200513303A</td>
      <td>TW-200513303-A</td>
      <td>Campaign doll</td>
      <td>Teamwell Technology Corp</td>
      <td>jian-zhong Cai</td>
      <td>2003-10-06</td>
      <td>2003-10-06</td>
      <td>2005-04-16</td>
      <td>NaN</td>
      <td>https://patents.google.com/patent/TW200513303A/en</td>
      <td>NaN</td>
      <td>TW200513303A</td>
      <td>['WO2003021374A3', 'CA2397397C', 'KR2003000537...</td>
      <td>\r\nThis invention relates to a campaign doll....</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>22573 rows × 14 columns</p>
</div>




```python
ids.to_csv("gp-q2-rich.csv")
```
