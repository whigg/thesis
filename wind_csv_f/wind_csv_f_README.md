# Fortranによるバイナリデータの読み込みとcsvファイルへの出力

## 環境
+ OS: macOS Sierra 10.12.6  
+ Compiler: GCC
+ Coding: Fortran77 (, Python3)


## 前提
バイナリデータが複数与えられていたとする．今回は，速度データ（`erainterim-1403_60`）．  
これをFortran77で読み込んで，バイナリではない数値の情報にcsvファイルとして出力・保存することを考える．   


## 読み込みデータの構造

```
erainterim-1403_60/  
	ecm030101.ads60  
	ecm030102.ads60  
	... (2084ファイル)  
```

各ファイルは列が（u,v,psl）の3列で，地図のグリッド数である145×145行分の数値が4byteのfloat型（fortranで言えばreal型）で入っている．  


## 準備
あらかじめ読み込むべきファイル名一覧（`ecm030101.ads60`など）をcsvファイルとして記録しておく．今回は，その名前を`output1.csv`とした．  
これは，pythonのファイル名取得用コードを実行することで簡単に得られる．（実行ファイル名：`wfname.py`）


## コード解説
<dl>
<dt>L.2-8</dt>
<dd>変数の定義．3行目のfilenameはいらないかも．<br>
u,v,psl: 速度u,vとpsl<br>
ret: csvに出力するための数値が入る配列<br>
count: ret用のポインタ<br>
fname: 出力先のファイル名（'~.csv'）<br>
x: output1.csvから読んだ入力ファイル名を格納<br>
data: 2084個分の入力ファイル名を格納</dd>
<dt>L.10-11</dt>
<dd>output1.csvをヘッダを飛ばして読む．</dd>
<dt>L.12-16</dt>
<dd>dataにファイル名を格納していく．</dd>
<dt>L.18-44</dt>
<dd>2084個のファイルに対し，以下を実行する：</dd>
<dt>L.21-22</dt>
<dd>まず，そのファイルを開く．</dd>
<dt>L.24-39</dt>
<dd>u,v,pslを出力用のretに格納．</dd>
<dt>L.41-43</dt>
<dd>fnameで指定したファイル名で書き出し．途中で書き出し用のsubroutineを呼び出した．</dd>
<dt>L.50-77</dt>
<dd>writecsvのsubroutine．引数は入力用の配列（vec）とファイル名（fname）．</dd>
<dt>L.65</dt>
<dd>書き込み先のファイルを開く．</dd>
<dt>L.66-75</dt>
<dd>書き込みを実行．write文を用いる．</dd>
<dt>L.79-90</dt>
<dd>csvへの書き込み時に余分なスペースが入るのを防ぐsubroutine．</dd>
</dl>


## 全体のファイル構造
※()は依存関係．これを見るとデータが重複して存在するが，3.を得たら
1.の中身を外に出せばいいと思う．（本当はFortranの方でパスを指定できればいいのだが，
調べるのがめんどくさかった…）  
※ここではデータは1つだけ入っている．

1. erainterim-1403_60  
	ecm030101.ads60  
	ecm030102.ads60  
	... (2084ファイル)
2. wfname.py (←1)
3. output1.csv (←2)
4. ecm030101.ads60, ecm030102.ads60, ...  
5. data2csv.f (←3,4)


## 出力
（上のファイル群と同じディレクトリに）

```
ecm030101.csv
ecm030102.csv
...
```

※本当はディレクトリを分けたかったが，fortran77でのパスの設定などを調べていなかった．


## 実行手順
MacのTerminalで，

```
$ gfortran data2csv.f
$ ./a.out
```


## 反省点・疑問点
+ `data2csv.f`で，複数回使う数字（145とか）をinteger型のparameterとして宣言すればよかったかも．  
ただし，それがsubroutineで継承されるのかは疑問（引数で渡せば良いのだが．）
+ 本当はFortran90でコーディングしたかったが，ファイルの読み込みがうまくいかず(L.21-22)，77のほうでうまくいったので全体を77に合わせた．
ファイルの読み込みでエラーが出なければ，90で書きたい．
+ もともとバイナリデータの読み込みはPythonでできたのだが，その時は調査不足で，研究室で主に使っているFortran77で渋々コーディングした結果となった．後日Pythonで同じことが比較的簡単にできてしまって，時間を返して欲しい☹️


## 参考文献（リンク）

1. [Fortran入門](http://nag-j.co.jp/fortran/index.html)
2. [Fortran77](http://www.geocities.jp/eyeofeconomyandhealth/homepage/renshuu1.html)
3. [csv読み込み](https://groups.google.com/forum/#!topic/comp.lang.fortran/1vL-UPZobqo)
4. [Fortranで文字列配列を使う方法](http://d.hatena.ne.jp/spadeAJ/20110209/1297230406)
5. [Fortran プログラムの構成](http://web.agr.ehime-u.ac.jp/~kishou/Lecture/atmosphere/Fortran90/Chap5.pdf)
6. [ファイル入出力](http://ax-b.com/FPR1_2014/class601/slides/140607.09.array_file.pdf)

ほか多数．

