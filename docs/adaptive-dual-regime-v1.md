# Adaptive Dual-Regime v1

`adaptive_dual_regime_v1` は、月次平均リターンの改善可能性を検証するためのペーパー専用戦略です。月4%を保証するものではありません。

## 安全境界

- `paper` または `mt5_paper` でのみ新規エントリー可能
- `oanda_practice` / `oanda_live` では新規エントリーを拒否
- 1日最大1件の新規エントリー
- 同時保有は最大2件
- 月初NAVから5%下落すると月内の新規エントリー停止
- 最高NAVから4%下落でリスク半減
- 7%下落で1取引リスクを最大0.25%に制限
- 10%下落で新規エントリー停止
- 同一通貨を含むポジションの予定損失はNAVの0.8%まで
- 合計予定損失はNAVの1.3%まで
- 実効レバレッジは最大3倍

## 戦略概要

毎日、対象10通貨ペアの日足をまとめて評価します。

1. 5日・20日・60日のボラティリティ調整済みモメンタムから通貨強弱を算出
2. ADX、EMA、ボリンジャーバンド幅からトレンド・レンジ・判別困難に分類
3. トレンドでは20日ブレイクまたは押し目・戻りを評価
4. レンジではZスコア1.8以上の行き過ぎからの反転を評価
5. 候補を0〜100点で採点し、最高点の1件だけを選択
6. スコア、相場変動、ドローダウンに応じて取引リスクを調整

スコア別の基準リスクは次のとおりです。

| スコア | 基準リスク |
|---|---:|
| 90以上 | NAVの0.80% |
| 82〜89 | NAVの0.65% |
| 75〜81 | NAVの0.40% |
| 75未満 | 取引なし |

レンジ取引は最大0.50%に制限します。現在ATRが過去60日の中央値より高い場合は、リスクを自動的に縮小します。

## 有効化

まず既存の`.env`をバックアップし、次を設定します。

```dotenv
BROKER_MODE=mt5_paper
TRADING_ARMED=false
STRATEGY_PROFILE=adaptive_dual_regime_v1
MARKET_DATA_CANDLE_COUNT=3200
MT5_INSTRUMENTS=USDJPY,EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,EURJPY,GBPJPY,AUDJPY
```

初回は`TRADING_ARMED=false`のまま実行します。

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/run?force=true"
```

`/api/status`で次を確認します。

- `strategy_profile` が `adaptive_dual_regime_v1`
- `market_data_candle_count` が `3200`
- 10通貨ペアが取得できる
- `currency_strength` が実行結果へ出力される
- エラーがない

確認後も、ペーパー注文を記録する場合だけ`TRADING_ARMED=true`へ変更します。MT5への実注文は送信されません。

## 重要指標日の手動ブロック

経済カレンダーの自動連携はこの変更に含まれていません。主要中銀会合、CPI、雇用統計、GDP速報などの日は、対象通貨を手動で設定します。

```dotenv
ENTRY_BLOCKED_CURRENCIES=USD,JPY
```

空に戻す場合は次のとおりです。

```dotenv
ENTRY_BLOCKED_CURRENCIES=
```

## 検証基準

実口座対応を検討する前に、少なくとも次を満たす必要があります。

- 2010年以降を含むバックテスト
- 取引1,000件以上
- スプレッド、スリッページ、スワップを控除
- プロフィットファクター1.30以上
- 純期待値0.30R以上
- プラス月65%以上
- 最大ドローダウン15%以下
- 3か月以上のMT5ペーパーフォワードテスト

基準を満たさない場合は、リスクを上げず、パラメーターの過剰最適化も行いません。
