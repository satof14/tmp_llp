# Deep Learning研究コード: Learning From Label Proportions (LLP) with Attention

## 概要
本研究では、Learning From Label Proportions (LLP)にAttentionメカニズムを導入した新しい手法を提案します。従来のLLPではインスタンスレベルの予測をバッグレベルに集約していましたが、提案手法は直接バッグレベルの予測を出力するTransformerベースのアーキテクチャを採用しています。

## 技術仕様

### 言語・フレームワーク
- 言語: Python
- フレームワーク: PyTorch

### データセット
- データセット: CIFAR-10
- 前処理: 各画像をパッチに分割（ViT方式）
- バッグサイズ: 可変（学習時：複数画像、推論時：1画像）

## アーキテクチャ詳細

### 入力形式
```
学習時: 
  - 入力: バッグ = {画像1, 画像2, ..., 画像N}
  - 教師データ: バッグのラベル割合 (例: [0.3, 0.5, 0.2] for 3クラス)
推論時: 単一画像
```

### モデル構成
1. パッチ化: 各画像をpatch_size × patch_sizeのパッチに分割（Linear Projectionを使用）
2. トークン化: 
   - BAG_CLSトークン（バッグレベル分類用）
   - 各画像のパッチトークン
3. Encoder: Global Attention + Local Attentionの交互構造（L回繰り返し）
4. MLP Head: BAG_CLSトークンからバッグレベル予測確率を出力

### Attention詳細
- Global Attention: すべてのトークン（BAG_CLS + 全画像のパッチ）間の関係性を学習
- Local Attention: 各画像内のパッチ間の関係性を学習

### 損失関数
```python
# Proportion Loss
# predicted_proportions: モデルが予測したバッグのラベル割合
# ground_truth_proportions: 実際のバッグのラベル割合（教師データ）
loss = CrossEntropyLoss(predicted_proportions, ground_truth_proportions)
```

### 学習・推論設定
- 学習時: バッグ（複数画像）を入力し、バッグのラベル割合（教師データ）を用いてバッグレベル予測を学習
- 推論時: 単一画像を入力し、同一モデルでインスタンスレベル分類を実行
- 評価指標: インスタンスレベル分類精度

## 特徴・利点
1. End-to-End学習: インスタンス→バッグの集約過程を省略
2. 可変入力: Transformerベースで任意のバッグサイズに対応
3. 統一アーキテクチャ: 学習と推論で同じモデル構造
4. 階層的Attention: 画像間（Global）と画像内（Local）の両方をモデル化

## 実装上の注意点
1. モデルサイズ: CIFAR-10に適したサイズに調整
3. 位置エンコーディング: 各画像内でのパッチ位置をエンコード。また、BAG_CLSトークンとその他のトークンを識別する情報もエンコード

## 最低限必要な引数
- patch_size：画像をpatch_size x patch_sizeにパッチ化
- embed_dim: トークンの埋め込み次元数
- num_heads: Attentionのヘッド数
- L：Encoder内でAttentionを繰り返す回数
- bag_size：学習時のバッグサイズ（一定）
- mini_batch_size：バッグレベルのバッチサイズ（推奨：2）
- num_classes：10（CIFAR-10）
- epochs：エポック数
- learning_rate：学習率

## データセットの位置
```
llp_attention/
└──
```