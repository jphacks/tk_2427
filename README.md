# Shopiris（しょぷりす）
![Colorful Planner Mobile App Promotion Instagram Post](https://github.com/user-attachments/assets/e3f047c4-1e6e-4c53-bda6-081130d55973)


## デモ動画
[こちら](https://youtu.be/Jx81Q2Q_JAw)からデモ動画をご覧いただけます．

## 開発メンバー
- [岡山　慎吾](https://github.com/shin5trp)
- [清　恵人](https://github.com/SeiKdesu)


## 目次
- [製品概要](#製品概要)
  - [開発背景](#開発背景)
  - [対象ユーザ](#対象ユーザ)
  - [特徴](#特徴)
    - [特徴1：発話を補助しすぎません！](#特徴1発話を補助しすぎません)
    - [特徴2：発話を視覚的に補助します！](#特徴2発話を視覚的に補助します)
    - [特長3：速度と精度を両立しています！](#特長3速度と精度を両立しています)
- [製品説明](#製品説明)
  - [iOSアプリ単体で使用する方法](#iosアプリ単体で使用する方法)
  - [iOSアプリをARデバイスと組み合わせて使用する方法](#iosアプリをarデバイスと組み合わせて使用する方法)
- [解決できること](#解決できること)
- [今後の展望](#今後の展望)
  - [展望1：対話内容に基づく予測単語の精度向上](#展望1対話内容に基づく予測単語の精度向上)
  - [展望2：UIやその他の機能の拡充](#展望2uiやその他の機能の拡充)
  - [展望3：Andoroid版の開発](#展望3andoroid版の開発)
- [注力したこと](#注力したこと)
  - [ポイント1：予測機能の実装](#ポイント1予測機能の実装)
  - [ポイント2：新たな情報源の導入](#ポイント2新たな情報源の導入)
  - [ポイント3：視覚的に邪魔になる情報の削除](#ポイント3視覚的に邪魔になる情報の削除)
- [開発技術](#開発技術)
  - [活用した技術](#活用した技術)
    - [API・データ](#api・データ)
    - [フレームワーク・ライブラリ・モジュール](#フレームワーク・ライブラリ・モジュール)
    - [デバイス](#デバイス)
  - [独自技術](#独自技術)
    - [ハッカソンで開発した独自機能・技術](#ハッカソンで開発した独自機能・技術)
      - [音声認識結果の保存機能](#音声認識結果の保存機能)
  - [製品に取り入れた研究内容](#製品に取り入れた研究内容)

## 製品概要
**「Shopping X Tech」**：視覚障害者が **実**店舗での買い物をより楽にできるアプリ

### 開発背景
視覚障害の方は人の情報である90%がないため大変困難です。そこで視覚障害向けの歩行を支援するアプリは開発されています。
しかし買い物を支援するアプリはないことに目を向けました。それはスーパーで視覚障害者の方を見たことがないように、商品を買うには値段などの情報や欲しい商品がどこにあるかといった情報が必要だからです。
**今見えているものを伝える歩行支援アプリではなく、欲しい商品の場所まで誘導してあげることが必要です。**
すでに買い物を補助するアプリは存在しています。
(https://www.asahi.com/articles/ASM2W3W7BM2WULBJ008.html?msockid=09c1f4de28566de31ec5e1db2c566f46)

しかしこのアプリでは**すでに登録した情報**をもとに支援します。これではどの店舗でも支援することは不可能です。

私たちはこの問題に着目し
- どの**実**店舗でも買い物を支援することのできる
- **音声**のみで支援することができる

そんなアプリ開発を目指しました．

### 対象ユーザ
- 日本人
- 視覚障害者
- スーパーなどで買いたい商品が決まっているユーザー

### 特長
#### 特徴1:どの店舗でも対応できること！
**物体検出技術** を用いて、**実**店舗で買いたい商品を買うことができる。
商品の情報をあらかじめ登録しておく必要がなくなります。視覚障害者の方のために全国のスーパーが商品情報を登録するのはとても現実的な改善策ではありません。どの店舗でも対応することができるようにします。
また、商品の場所や商品の在庫などが変わっていく中でも商品の場所まで案内することができる。

#### 特徴2：セキュリティ面が安全
すべてon deviceで動作しておりAPIなどでネットワークとの通信接続は一切行っていません。動画で常に撮影し続けるため、その動画のデータが漏れてまったら生活そのものがすべて筒抜け状態になってしまいます。なので安心して使えるようにインターネット通信は一切行わずに作成しました。また、データの通信速度制限によって支援できないことを避けたりすることもできます。

#### 特長3：自作モデル？


## 製品説明
1. 買いたい商品を音声で入力する。（複数商品入力することができる。）
2. 買いたい商品を選択したら「確定」と話す
3. 動画撮影の画面へ遷移する
4. 買いたい商品が近くにあれば音声によって左側にりんごがあるといった支援を行い、商品へ近づける
5. ![image](https://github.com/user-attachments/assets/c51418c1-ded0-46e4-8f02-51ac74a40ead)

6. 商品の前まで近づいたら、商品のポップから値段などの商品情報を音声で伝える
7. かごに商品を入れたら、3本指で画面をタップする
8. リストに購入したい商品が残っていれば手順の３に移動し、商品を探す    
  
<img width="20%" alt="音声認識時のスマホ画面" src="https://github.com/jphacks/NG_2305/assets/109562639/09dbf542-187a-4ac4-8048-79c9c90ea893">　<img width="20%" alt="音声認識時のスマホ画面" src="https://github.com/jphacks/NG_2305/assets/109562639/e3c547d5-c232-4ea0-9747-bd8053f56f73">　<img width="20%" alt="音声認識時のスマホ画面" src="https://github.com/jphacks/NG_2305/assets/109562639/e24ccf19-6aa8-4fb5-acda-ee41bdeb986b">





## 解決できること
ほしい商品をどのような**実**店舗でより早く見つけるだけではなく、値段などの情報もわかるようにすることで、より快適な買い物をすることができる。
スーパーの方が商品の登録をする必要がなくなる。

## 今後の展望
### 展望1：単眼深度推定により商品の位置も伝えるようにする
商品の方向を伝えることはできているが、商品の距離などを伝えることはできていないため、何メートル先にあるといったことを伝えるようにする。
  
### 展望2：商品の支援する順番
スーパーは大体野菜→生肉→お菓子→冷凍食品→飲料→お惣菜→パン類などある程度決まった場所に配置されているので、ある程度の予測モデルを用いた支援．

### 展望3：クラス数の増加
商品を選択することのできるクラス数に縛りがあるためfunetuningを行なって選択可能なクラス数を増やしていく。

## 注力したこと
### ポイント1：画像認識（YOLOの実装）
どのお店にも対応できるように画像認識を用いた。

### ポイント2：on device での実装
セキュリティ面の問題などから全てon deviceで完結することから支援に対してどのような環境でも支援することが可能になる。

### ポイント3：すべて音声で完結していること
視覚障害者のため、音声での完結。３本指でタップするなどのアプリ起動からアプリ終了まで全て視覚情報を一切用いないこと。


## 開発技術
<img width="100%" alt="開発技術の概要図" src="https://github.com/jphacks/NG_2305/assets/103105513/6fc83e97-27b3-4802-b044-31bab11f751b">


### 活用した技術
#### CoreML 
- YOLOv3学習済みモデル
#### Swift
- AVFoundation
- Vision



#### デバイス
- iphone (ios 18.0.1)
    - ios デバイス14.0以上で実装可能

### 独自技術
#### ハッカソンで開発した独自機能・技術
##### 音声認識結果の保存機能
commit_id: [5674e3c7c945d6a1c6ec8a8ade41fcb9ef19ea1d](https://github.com/jphacks/NG_2305/commit/5674e3c7c945d6a1c6ec8a8ade41fcb9ef19ea1d)

音声認識自体はSpeechフレームワークを用いて実装し，音声認識結果を保存する追加機能を独自に実装しました．  
SFSpeechRecognizerを用いて音声をテキストに起こす際に，以下の3つの問題が存在していました．
- Speech frameworkの仕様により音声認識の途中結果がいくつも出力される
- 時間経過により認識された文章が削除されると認識結果が重複して画面に表示される
- 発話が少しでも止まるとそれ以前の認識結果が失われる

これらの問題を解決するために，以下の機能を実装しました．
- 音声認識の途中結果をバッファリング
- 認識完了を検知したら最終結果を画面に出力
- 時間経過で認識された文章が削除されないように認識結果を保存

### 製品に取り入れた研究内容
なし


# Shopiris（しょぷりす）

[![IMAGE ALT TEXT HERE](https://jphacks.com/wp-content/uploads/2024/07/JPHACKS2024_ogp.jpg)](https://www.youtube.com/watch?v=DZXUkEj-CSI)

## 製品概要
### 背景(製品開発のきっかけ、課題等）
視覚障害者の歩行を支援するアプリは既存にあるが、買いたい商品を買うことのできる支援アプリはない。スーパーにたどり着いたとしても解体商品がどこにあるかがわからない。
### 製品説明（具体的な製品の説明）
1. 買いたい商品を音声で入力する。（複数商品入力することができる。）
2. 買いたい商品を選択したら「確定」と話す。
3. 動画撮影の画面へ遷移する。
4. 買いたい商品が近くにあれば音声によって左側にりんごがあるといった支援を行い、商品へ近づける。
5. 商品の前まで近づいたら、商品のポップから値段などの商品情報を音声で伝える。
6. かごに商品を入れたら、3本指で画面をタップする。
7. リストに購入したい商品が残っていれば手順の３に移動し、商品を探す。
### 特長
#### 1. 特長1
商品の位置まで誘導するだけでなく、商品の情報を音声で伝えられること
#### 2. 特長2
on deviceですべてネットワークとの通信をせずに行うことができるためセキュリティの問題に強い
#### 3. 特長3
自作モデル開発。画像の認識可能クラスを増やすためにfine-tuning

### 解決出来ること
視覚障害者が１人で買いたい商品が買うことができることや、その商品の情報が視覚障害者に伝えられ楽しく買い物をすることができる。
### 今後の展望
- 単眼深度推定により商品の位置も伝えるようにする。
- スーパーは大体野菜→生肉→お菓子→冷凍食品→飲料→お惣菜→パン類などある程度決まった場所に配置されているので、ある程度の予測モデルを用いた支援
- 毎日同じスーパーに通うのであれば日々のデータからどこに何があるかを学習し最適なルートで欲しい商品を手に入れられるようにする。
- クラス数に縛りがあるためfunetuningを行なって選択可能なクラス数を増やしていく。
- 「確定」の後、商品の追加をしたい時に商品を選択し直したくなった場合の戻る機能の追加。誤って３本タップした時の戻る機能など。
### 注力したこと（こだわり等）
* どのお店にも対応できるように画像認識を用いた。（既存に購入可能なデータセットを用意しておくのではなく）
* 全てon deviceで完結することから支援に対してどのような環境でも支援することが可能になる。
* 視覚障害者のため、音声での完結。３本指でタップするなどのアプリ起動からアプリ終了まで全て視覚情報を一切用いないこと。
* 

## 開発技術

- YOLOを用いた画像認識
- 購入したい商品の音声認識
- swift?
### 活用した技術
技術がここにくる？
CoreMLの画像認識技術
#### API・データ
* 食べ物の学習データ（画像）
* 

#### フレームワーク・ライブラリ・モジュール
* swift
* coreML
* YOLO

#### デバイス
* ios app
* 

### 独自技術
#### ハッカソンで開発した独自機能・技術
* 独自で開発したものの内容をこちらに記載してください
* 特に力を入れた部分をファイルリンク、またはcommit_idを記載してください。
