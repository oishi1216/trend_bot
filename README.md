# FX Trend Bot — OANDA REST API版MVP

日足の低レバレッジ・トレンドフォロー戦略を、Python・FastAPI・OANDA v20 REST APIで評価／執行するためのMVPです。

> **重要な訂正（2026-07-04）**  
> このリポジトリは **MT5用EAではありません**。また、OANDA Japanのデモ口座を作成しただけでは、このアプリに必要なREST APIトークンを利用できるとは限りません。OANDA Japanで今すぐデモ自動売買を始める場合は、まず **MT5＋MQL5 EA** を使う構成が現実的です。

## 1. 実装している戦略

- 200日EMAの方向フィルター
- 過去55日高値／安値のブレイクで新規
- 20日反対チャネルで手仕舞い
- 初期ストップは20日ATRの2倍
- 1取引リスク0.25%、合計予定リスク0.75%
- グロス実効レバレッジ上限2倍
- 8%ドローダウンでリスク半減、12%で新規停止
- ナンピン、マーチンゲール、損切り拡大なし

> これは利益保証ソフトではありません。急変、窓開け、通信障害、API障害、スリッページ、価格配信停止などで、予定した損失を超えることがあります。

## 2. OANDA Japanでの位置づけ

OANDA Japanのデモ口座では、FXについてMT5およびfxTrade／TradingViewを利用できます。ただし、次の3つは別物です。

| 種類 | 用途 | このリポジトリからの接続 |
|---|---|---|
| MT5デモ口座 | MT5上で裁量取引・EA・ストラテジーテスト | **未対応** |
| fxTrade／TradingView | チャート分析・手動発注 | **直接接続しない** |
| OANDA v20 REST API | Pythonなど外部プログラムから価格取得・注文 | **対応対象** |

### MT5のログイン情報はAPI認証情報ではない

`OANDA_ACCOUNT_ID` に入力するのは、REST APIで利用可能なv20／fxTrade口座IDです。MT5のログインIDを入力しても接続できません。

### OANDA JapanのAPI利用条件

OANDA Japanの公式案内では、REST APIの利用には少なくとも以下が必要です。

- 会員ステータスがGold以上
- NYサーバーの取引コースがプロコース
- NYサーバー口座残高が25万円以上
- API契約への同意とパーソナルトークンの発行

条件や対象口座は変更される可能性があります。利用前に公式ページで最新条件を確認してください。

- [OANDA Japan API案内](https://www.oanda.jp/platform/api)
- [APIトークンの発行条件と手順](https://www.oanda.jp/lab-education/api/usage/rest_api_activation_procedure/)
- [MT4/MT5デモ口座設定](https://www.oanda.jp/platform/mt4/flow/demo)

## 3. 推奨する進め方

### ルートA：MT5デモで検証する（現在の推奨）

OANDA Japanのデモ口座を使い、今回の戦略をMQL5のEAとして別途実装します。

```text
OANDA Japan MT5デモ口座
        ↓
       MT5
        ↓
    MQL5 EA
        ↓
ストラテジーテスター／フォワードテスト
```

このリポジトリにはMQL5 EAは含まれていません。Python版とは別実装が必要です。

### ルートB：REST API版を使う

API利用条件を満たし、OANDAのAPI画面から有効なパーソナルトークンと口座IDを取得できた場合に、このリポジトリを利用します。

```text
OANDA v20 REST API
        ↕
Python / FastAPI
        ↕
SQLite・管理画面
```

最初は`paper`モードでシグナルと仮想損益だけを検証し、実注文は送らないでください。

### ルートC：TradingViewから自動化する

TradingViewとOANDA口座を接続しただけでは、Pine Scriptのストラテジーを直接自動売買にはできません。自動化には一般に次の構成が必要です。

```text
TradingViewアラート
        ↓ Webhook
外部Pythonアプリ
        ↓ REST API
OANDA
```

このMVPはTradingView Webhookの受信機能を実装していません。

## 4. 現在の対応状況

| 機能 | 状態 |
|---|---|
| OANDA v20 REST APIのローソク足取得 | 実装済み |
| REST API口座情報・価格取得 | 実装済み |
| ローカルのペーパートレード | 実装済み |
| OANDA practice APIへの注文 | 実装済み。ただし対応するpractice認証情報が必要 |
| OANDA live APIへの注文 | 実装済み。実口座投入前の追加検証が必要 |
| MT5への接続 | 未実装 |
| MQL5 EA | 未実装 |
| TradingView Webhook受信 | 未実装 |
| CSVだけを使うオフラインバックテスト | 未実装 |

## 5. モードに関する重要事項

`.env`では次の3モードを指定できます。

```dotenv
# paper | oanda_practice | oanda_live
BROKER_MODE=paper
```

### `paper`

注文と損益はSQLite内で模擬します。ただし、現在の実装は価格取得、口座通貨換算、時価評価にOANDA REST APIを使います。

そのため、**paperモードでも有効なREST APIの口座IDとトークンが必要です**。MT5デモ口座のログイン情報だけでは動作しません。

### `oanda_practice`

`https://api-fxpractice.oanda.com`へ注文します。

このモードは、OANDA v20 practice環境用として発行された認証情報がある場合だけ使用してください。**OANDA JapanのMT5デモ口座と同じものではありません**。MT5デモの口座IDやパスワードを設定しても接続できません。

### `oanda_live`

`https://api-fxtrade.oanda.com`へ実注文を送ります。OANDA Japanで利用する場合は、API利用条件を満たした口座と発行済みパーソナルトークンが必要です。

二重ロックとして、次の両方が設定されていなければ新規発注しません。

```dotenv
BROKER_MODE=oanda_live
TRADING_ARMED=true
ALLOW_LIVE_TRADING=YES_I_ACCEPT_THE_RISK
```

十分なバックテスト、フォワードテスト、障害試験を終えるまで解除しないでください。

## 6. REST APIを利用できる場合の起動方法

APIを利用できない場合は、この章の手順では起動できません。MT5 EA版を用意するか、市場データ取得部分を別のデータソースに置き換える必要があります。

### 6.1 環境ファイルを作る

```bash
cp .env.example .env
```

`.env`へREST API用の情報を入力します。

```dotenv
BROKER_MODE=paper
TRADING_ARMED=false

# MT5ログインIDではなく、REST APIで利用できる口座ID
OANDA_ACCOUNT_ID=...

# OANDAのAPIアクセス管理画面で発行したパーソナルトークン
OANDA_API_TOKEN=...

ACCOUNT_HOME_CURRENCY=JPY
```

### 6.2 Dockerで起動する

```bash
docker compose up --build -d
```

ブラウザーで次を開きます。

```text
http://127.0.0.1:8000
```

最初は`TRADING_ARMED=false`のまま実行し、シグナルとエラーだけ確認してください。

### 6.3 手動実行する

```bash
curl -X POST http://127.0.0.1:8000/api/run
```

`force=true`は同じ足を再評価する開発用オプションです。注文重複の原因になり得るため、本番では使用しないでください。

## 7. 実行タイミング

月曜〜金曜の22:15 UTC以降に1日1回、自動実行します。OANDAの日足は`America/New_York`の17時で揃え、未完成足を除外します。同じ完成足はSQLiteで重複処理しません。

日本では米国の夏時間によって実行時刻の見え方が変わります。Dockerホストの時刻とタイムゾーン設定も確認してください。

## 8. 安全設計

- 新規注文は`TRADING_ARMED`で停止可能。既存ポジションの出口判定は継続します。
- liveモードでは追加の文字列ロックがあります。
- OANDA側に存在し、ローカルDBにないポジションを検出すると、その通貨の新規注文をブロックします。
- ローカルポジションがOANDA側から消えた場合、ブローカー側ストップ等で閉じた可能性があるとして警告を記録します。
- Web画面に認証機能はありません。Docker Composeはlocalhostだけに公開します。外部公開しないでください。
- APIトークンをGitへコミットしないでください。
- MT5による手動取引や別EAと同じ口座を混在させないでください。

## 9. テスト

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 10. 本番化前に追加すべきもの

1. 15年以上・4通貨合算のバックテストとウォークフォワード検証
2. スプレッド、日次スワップ、週末ギャップ、注文拒否を含むシミュレーション
3. Slack／メール通知、死活監視、ログ集中管理
4. ブローカー取引履歴を使った完全なポジション照合
5. API再試行の冪等性キーと注文重複防止
6. DBバックアップ、秘密情報管理、アクセス認証
7. ブローカーごとの最小取引単位・価格精度・ストップ最小距離の取得と検証
8. OANDA Japan口座でのエンドポイント、口座ID形式、注文仕様の実地確認
9. MT5を使う場合は、同等ロジックのMQL5 EAとストラテジーテスト

## 11. 重要な実装上の注意

このMVPのローカルDBは「この戦略が建てたポジション」の台帳です。手動売買や別のEAと同じ口座を混在させないでください。

注文送信後に通信が切れた場合など、実際には約定したのにレスポンスを受け取れない障害を完全には処理していません。また、OANDA JapanのMT5デモ口座を、このPythonアプリへ直接接続する機能はありません。

このリポジトリは、実口座ですぐ稼働させる完成品ではなく、REST APIを利用できる環境で戦略とリスク管理を検証するための出発点です。
