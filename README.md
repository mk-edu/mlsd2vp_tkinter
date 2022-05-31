# M-LSDを用いたvpの検出をGUI表示
mlsd2vp.ymlファイルを用いて仮想環境を再構築

`$ conda env create -n 新しい環境名 -f mlsd2vp.yml`

[ここ](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)からデータセットを取得しmlsd2vp_tkinterの下に配置

```
$ conda activate mlsd2vp
$ python mlsd_tkinter.py
```
## tkinterを使ったGUI

左上のドロップダウンリストから**YorkUrbanDB**を選択し，**select image**を押すと，ダウンロードしたデータセットの画像を用いることができます．

画像を選択すると，左のキャンバスに表示されるので**run M-LSD**ボタンを押してください．

Ground Truthの消失点，検出された消失点，それぞれの消失点に対応する線を画像にしたものが右のフレーム内に表示されます．

**score**, **dist**, **length**のパラメータの値によって出力結果が異なります．自由に変えてみてください．

**reset value**ボタンは3つのパラメータを初期値にリセットします．

**SAVE Image**ボタンは画像を保存します．

**clear result**ボタンは右の出力結果を削除します．
 
**All clear**ボタンはすべての入出力を削除します．

![sample](https://github.com/mk-edu/mlsd2vp_tkinter/blob/image/sample.png)

## もともとのVP検出アルゴリズム
[Python + OpenCV implementation of the Vanishing Point algorithm](https://github.com/rayryeng/XiaohuLuVPDetection)

こちらのアルゴリズムは線分検出に[LSD](https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf)を用いていました

## M-LSDとは？
[M-LSD: Towards Light-weight and Real-time Line Segment Detection](https://github.com/navervision/mlsd)

深層学習によってリソースが制限された環境におけるリアルタイムかつ軽量な線分検出器らしいです．

従来のLSDと比べて，必要のない線が検出されることが少ないです．そのため，建物の枠組みだけを線で検出したいときはLSDよりもM-LSDのほうがいいのではないでしょうか． 
