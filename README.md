# Shopiris（しょぷりす）
![Colorful Planner Mobile App Promotion Instagram Post](https://github.com/user-attachments/assets/e3f047c4-1e6e-4c53-bda6-081130d55973)


## デモ動画
[こちら](https://youtu.be/Jx81Q2Q_JAw)からデモ動画をご覧いただけます．

## 開発メンバー
- [岡山　慎吾](https://github.com/shin5trp)
- [清　恵人](https://github.com/SeiKdesu)


## 目次
- [製品概要](#製品概要)
  - [背景](#背景)
  - [対象ユーザ](#対象ユーザ)

- [製品説明](#製品説明)
- [特徴](#特徴)
  - [特徴1:どの店舗でも対応できること！](#特徴1:どの店舗でも対応できること！)
  - [特徴2:セキュリティ面が安全!](#特徴2：セキュリティ面が安全)
- [解決できること](#解決できること)
- [今後の展望](#今後の展望)
  - [展望1：単眼深度推定により商品の位置も伝えるようにする](#展望1：単眼深度推定により商品の位置も伝えるようにする)
  - [展望2：クラス数の増加](#展望2：クラス数の増加)
- [注力したこと（こだわり等）](#注力したこと（こだわり等）)
  - [ポイント1：画像認識（YOLOの実装）](#ポイント1：画像認識（YOLOの実装）)
  - [ポイント2：on device での実装](#ポイント2：on device での実装)
  - [ポイント3：すべて音声で完結していること](#ポイント3：すべて音声で完結していること)
- [開発技術](#開発技術)
  - [活用した技術](#活用した技術)
    - [CoreML ](#CoreML )
    - [Swift](#Swift)
    - [デバイス](#デバイス)

## 製品概要
**「Shopping X Tech」**：視覚障害者が **実**店舗での買い物をより楽にできるアプリ

### 背景(製品開発のきっかけ、課題等)
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

### 特徴
#### 特徴1:どの店舗でも対応できること！
**物体検出技術** を用いて、**実**店舗で買いたい商品を買うことができる。
商品の情報をあらかじめ登録しておく必要がなくなります。視覚障害者の方のために全国のスーパーが商品情報を登録するのはとても現実的な改善策ではありません。どの店舗でも対応することができるようにします。
また、商品の場所や商品の在庫などが変わっていく中でも商品の場所まで案内することができる。

#### 特徴2：セキュリティ面が安全
すべてon deviceで動作しておりAPIなどでネットワークとの通信接続は一切行っていません。動画で常に撮影し続けるため、その動画のデータが漏れてまったら生活そのものがすべて筒抜け状態になってしまいます。なので安心して使えるようにインターネット通信は一切行わずに作成しました。また、データの通信速度制限によって支援できないことを避けたりすることもできます。




## 製品説明
1. 買いたい商品を音声で入力する。（複数商品入力することができる。）
2. 買いたい商品を選択したら「確定」と話す
3. 動画撮影の画面へ遷移する
4. 買いたい商品が近くにあれば音声によって左側にりんごがあるといった支援を行い、商品へ近づける
5. ![image](https://github.com/user-attachments/assets/c51418c1-ded0-46e4-8f02-51ac74a40ead)

6. 商品の前まで近づいたら、商品のポップから値段などの商品情報を音声で伝える
7. かごに商品を入れたら、3本指で画面をタップする
8. リストに購入したい商品が残っていれば手順の３に移動し、商品を探す    
  





## 解決できること
ほしい商品をどのような**実**店舗でより早く見つけるだけではなく、値段などの情報もわかるようにすることで、より快適な買い物をすることができる。
スーパーの方が商品の登録をする必要がなくなる。

## 今後の展望
### 展望1：単眼深度推定により商品の位置も伝えるようにする
商品の方向を伝えることはできているが、商品の距離などを伝えることはできていないため、何メートル先にあるといったことを伝えるようにする。
  
### 展望2：クラス数の増加
商品を選択することのできるクラス数に縛りがあるためfunetuningを行なって選択可能なクラス数を増やしていく。

![image](https://github.com/user-attachments/assets/5ee67f87-be57-4ee5-b45f-38dcb1b921d5)

## 注力したこと
### ポイント1：画像認識（YOLOの実装）
どのお店にも対応できるように画像認識を用いた。

### ポイント2：on device での実装
セキュリティ面の問題などから全てon deviceで完結することから支援に対してどのような環境でも支援することが可能になる。

### ポイント3：すべて音声で完結していること
視覚障害者のため、音声での完結。３本指でタップするなどのアプリ起動からアプリ終了まで全て視覚情報を一切用いないこと。


## 開発技術
![イエロー　シンプル　イラスト　応募方法　フロー　Twitter投稿](https://github.com/user-attachments/assets/46f07b63-3339-4883-8fdc-dcc1a48aa1fc)



### 活用した技術
#### CoreML 
- YOLOv3学習済みモデル
#### Swift
- AVFoundation
- Vision



#### デバイス
- iphone (ios 18.0.1)
    - ios デバイス14.0以上で実装可能
