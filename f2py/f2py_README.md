# バイナリデータをf2pyでPythonへインポートする

## 環境など
+ `wind_csv_f`と同じ．
+ バイナリデータも同じ`ecm030101.ads60`を使用


## 実行
MacのTerminalで，

```
$ f2py -c -m mystring loaddata.f
```

次に，Pythonインタプリンタで，Pythonを起動し，インポートする：

```
$ python
$ import mystring

#データのロード
>>> mystring.loaddata(filename) ex. "ecm030101.ads60"
# -7.00242853      -1.04627931       1020.24927    
# -7.16496706      0.811227083       1020.24786 ...
```

名前'mystring'は，自分の好みにできる．（今回は参考にしたサイトのものを踏襲してしまった）


## 参考リンク
1. [Using F2PY bindings in Python](https://docs.scipy.org/doc/numpy-dev/f2py/python-usage.html)
2. [f2pyを使ってfortranでpythonのモジュールを書く](https://qiita.com/airtoxin/items/b632f2b3f219610f3990)