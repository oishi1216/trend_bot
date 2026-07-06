# FX Trend Bot — OANDA REST API / MT5 paper MVP

日足の低レバレッジ・トレンドフォロー戦略を、Python・FastAPIで評価／ペーパー運用するためのMVPです。

> **重要**  
> これは利益保証ソフトではありません。急変、窓開け、通信障害、API障害、スリッページ、価格配信停止などで、予定した損失を超えることがあります。

## 1. 実装している戦略

- 200日EMAの方向フィルター
- 過去55日高値／安値のブレイクで新規
- 20日反対チャネルで手仕舞い
- 初期ストップは20日ATRの2倍
- 1取引リスク0.25%、合計予定リスク0.75%
- グロス実効レバレッジ上限2倍
- 8%ドローダウンでリスク半減、12%で新規停止
- ナンピン、マーチンゲール、損切り拡大なし

## 2. OANDA Japanでの位置づけ

OANDA Japanのデモ口座では、FXについてMT5およびfxTrade／TradingViewを利用できます。ただし、次の3つは別物です。

| 種類 | 用途 | このリポジトリからの接続 |
|---|---|---|
| MT5デモ口座 | MT5上で裁量取引・EA・ストラテジーテスト | `mt5_paper`で価格取得のみ対応 |
| fxTrade／TradingView | チャート分析・手動発注 | 直接接続しない |
| OANDA v20 REST API | Pythonなど外部プログラムから価格取得・注文 | `paper` / `oanda_practice` / `oanda_live`で対応 |

### MT5のログイン情報はREST API認証情報ではない

`OANDA_ACCOUNT_ID` に入力するのは、REST APIで利用可能なv20／fxTrade口座IDです。MT5のログインIDを入力してもREST APIには接続できません。

### OANDA JapanのAPI利用条件

OANDA JapanのREST API利用には、会員ステータス、取引コース、口座残高、API契約などの条件があります。利用前に公式ページで最新条件を確認してください。

- [OANDA Japan API案内](https://www.oanda.jp/platform/api)
- [APIトークンの発行条件と手順](https://www.oanda.jp/lab-education/api/usage/rest_api_activation_procedure/)
- [MT4/MT5デモ口座設定](https://www.oanda.jp/platform/mt4/flow/demo)

## 3. 推奨する進め方

### ルートA：MT5ペーパーモードで検証する

OANDA REST APIが使えない場合は、まず `mt5_paper` を使います。

```text
OANDA Japan MT5デモ口座
        ↓
       MT5端末
        ↓
MetaTrader5 Python package
        ↓
Python / FastAPI
        ↓
SQLiteペーパー売買・OpenAIフィードバック
```

`mt5_paper` は、MT5から日足データと現在価格を取得します。ただし、**MT5へ注文は送りません**。注文・損益・ストップはSQLite内で模擬します。

### ルートB：REST API版を使う

API利用条件を満たし、OANDAのAPI画面から有効なパーソナルトークンと口座IDを取得できた場合に、REST API版を利用します。

```text
OANDA v20 REST API
        ↕
Python / FastAPI
        ↕
SQLite・管理画面
```

最初は `paper` モードでシグナルと仮想損益だけを検証し、実注文は送らないでください。

### ルートC：TradingViewから自動化する

TradingViewとOANDA口座を接続しただけでは、Pine Scriptのストラテジーを直接自動売買にはできません。自動化には一般に次の構成が必要です。

```text
TradingViewアラート
        ↓ Webhook
外部Pythonアプリ
        ↓ REST APIまたはMT5
OANDA
```

このMVPはTradingView Webhookの受信機能を実装していません。

## 4. 現在の対応状況

| 機能 | 状態 |
|---|---|
| OANDA v20 REST APIのローソク足取得 | 実装済み |
| REST API口座情報・価格取得 | 実装済み |
| ローカルのペーパートレード | 実装済み |
| MT5からの日足・現在価格取得 | `mt5_paper`で実装済み |
| MT5経由のペーパー売買 | 実装済み |
| MT5への実注文 | 未実装 |
| MQL5 EA | 未実装 |
| TradingView Webhook受信 | 未実装 |
| CSVだけを使うオフラインバックテスト | 未実装 |

## 5. モードに関する重要事項

`.env`では次の4モードを指定できます。

```dotenv
# paper | mt5_paper | oanda_practice | oanda_live
BROKER_MODE=mt5_paper
```

### `paper`

注文と損益はSQLite内で模擬します。ただし、価格取得、口座通貨換算、時価評価にOANDA REST APIを使います。

そのため、**paperモードでも有効なREST APIの口座IDとトークンが必要です**。MT5デモ口座のログイン情報だけでは動作しません。

### `mt5_paper`

MT5から価格データを取得し、注文と損益はSQLite内で模擬します。

- OANDA REST APIトークンは不要
- Windows上でOANDA MetaTrader 5を起動しておく必要あり
- MT5デモ口座へログイン済みである必要あり
- `TRADING_ARMED=true`にしてもMT5へ実注文は送りません
- 最新のMT5日足は未確定の可能性があるため、取得後に最後の1本を除外します

### `oanda_practice`

`https://api-fxpractice.oanda.com`へ注文します。

このモードは、OANDA v20 practice環境用として発行された認証情報がある場合だけ使用してください。**OANDA JapanのMT5デモ口座と同じものではありません**。

### `oanda_live`

`https://api-fxtrade.oanda.com`へ実注文を送ります。OANDA Japanで利用する場合は、API利用条件を満たした口座と発行済みパーソナルトークンが必要です。

二重ロックとして、次の両方が設定されていなければ新規発注しません。

```dotenv
BROKER_MODE=oanda_live
TRADING_ARMED=true
ALLOW_LIVE_TRADING=YES_I_ACCEPT_THE_RISK
```

十分なバックテスト、フォワードテスト、障害試験を終えるまで解除しないでください。

## 6. `mt5_paper` の起動方法

### 6.1 Python依存関係を入れる

通常依存をインストールします。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

MT5連携を使うWindows環境だけ、追加で次を入れます。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-mt5.txt
```

### 6.2 MT5を準備する

1. OANDA MetaTrader 5を起動
2. OANDA MT5デモ口座へログイン
3. 気配値に `USDJPY`、`EURUSD`、`GBPUSD`、`AUDUSD` が表示されることを確認
4. Pythonから `account_info` と日足データが取得できることを確認

### 6.3 `.env` を設定する

```powershell
Copy-Item .env.example .env
notepad .env
```

Windowsローカル実行では、`DB_PATH` とMT5設定を次のようにします。

```dotenv
DB_PATH=data/fxbot.sqlite3
BROKER_MODE=mt5_paper
TRADING_ARMED=false

MT5_TERMINAL_PATH=C:\Program Files\OANDA MetaTrader 5\terminal64.exe
MT5_INSTRUMENTS=USDJPY,EURUSD,GBPUSD,AUDUSD
MT5_TIMEOUT_MS=120000

OPENAI_FEEDBACK_ENABLED=true
OPENAI_FEEDBACK_ALLOW_LIVE=false
OPENAI_API_KEY=sk-...
```

`OPENAI_API_KEY`をGitへコミットしないでください。

### 6.4 アプリを起動する

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --env-file .env
```

ブラウザーで次を開きます。

```text
http://127.0.0.1:8000
```

別PowerShellから実行します。

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/run?force=true"
```

イベント確認：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/events?limit=20"
```

## 7. REST APIを利用できる場合の起動方法

APIを利用できない場合は、この章の手順では起動できません。`mt5_paper`を使うか、市場データ取得部分を別のデータソースに置き換える必要があります。

### 7.1 環境ファイルを作る

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

### 7.2 Dockerで起動する

```bash
docker compose up --build -d
```

ブラウザーで次を開きます。

```text
http://127.0.0.1:8000
```

最初は`TRADING_ARMED=false`のまま実行し、シグナルとエラーだけ確認してください。

## 8. 実行タイミング

月曜〜金曜の22:15 UTC以降に1日1回、自動実行します。OANDA REST APIの日足は`America/New_York`の17時で揃え、未完成足を除外します。MT5日足は保守的に最後の1本を除外します。同じ完成足はSQLiteで重複処理しません。

日本では米国の夏時間によって実行時刻の見え方が変わります。ホストの時刻とタイムゾーン設定も確認してください。

## 9. 安全設計

- 新規注文は`TRADING_ARMED`で停止可能。既存ポジションの出口判定は継続します。
- `mt5_paper`はMT5へ実注文を送りません。
- liveモードでは追加の文字列ロックがあります。
- OANDA側に存在し、ローカルDBにないポジションを検出すると、その通貨の新規注文をブロックします。
- ローカルポジションがOANDA側から消えた場合、ブローカー側ストップ等で閉じた可能性があるとして警告を記録します。
- Web画面に認証機能はありません。外部公開しないでください。
- APIトークンやMT5パスワードをGitへコミットしないでください。
- MT5による手動取引や別EAと同じ口座を混在させないでください。

## 10. テスト

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOSや通常CIでは、MT5追加依存を入れずに通常テストだけ実行できます。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 11. OpenAI APIへテスト結果を送る機能

エンジン実行後の集計結果をOpenAI APIへ送り、構造化された評価結果をSQLiteの`events`テーブルへ保存する機能が含まれます。初期状態では無効です。

### 送信される内容

- 最新の`engine_run`集計
- 選択された直近の運用イベント
- 現在の戦略パラメータとリスク上限

APIキー、OANDAアカウントID、認証情報、`raw`ブローカー応答、ブローカー取引IDは送信前に削除またはマスクされます。ただし、ログ設計を変更した場合は送信内容を再確認してください。

### 設定

`.env`へ追加します。

```dotenv
OPENAI_FEEDBACK_ENABLED=true
OPENAI_FEEDBACK_ALLOW_LIVE=false
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
OPENAI_FEEDBACK_EVENT_LIMIT=100
OPENAI_FEEDBACK_MIN_INTERVAL_HOURS=20
OPENAI_TIMEOUT_SECONDS=60
```

`OPENAI_API_KEY`をGitへコミットしないでください。実口座モードでは`OPENAI_FEEDBACK_ALLOW_LIVE=false`のままにすることを推奨します。

### 実行

通常のエンジン実行後、設定された最小間隔を超えていれば自動的に評価されます。

```bash
curl -X POST "http://127.0.0.1:8000/api/run"
```

直近のエンジン結果を強制的に再評価する場合：

```bash
curl -X POST "http://127.0.0.1:8000/api/feedback/run?force=true"
```

最新の評価結果：

```bash
curl "http://127.0.0.1:8000/api/feedback/latest"
```

この機能は分析結果を保存するだけで、コード変更、GitHub Push、Pull Request作成、取引パラメータ変更は行いません。それらは別ワークフローとして実装してください。

## 12. 本番化前に追加すべきもの

1. 15年以上・4通貨合算のバックテストとウォークフォワード検証
2. スプレッド、日次スワップ、週末ギャップ、注文拒否を含むシミュレーション
3. Slack／メール通知、死活監視、ログ集中管理
4. ブローカー取引履歴を使った完全なポジション照合
5. API再試行の冪等性キーと注文重複防止
6. DBバックアップ、秘密情報管理、アクセス認証
7. ブローカーごとの最小取引単位・価格精度・ストップ最小距離の取得と検証
8. OANDA Japan口座でのエンドポイント、口座ID形式、注文仕様の実地確認
9. MT5で実注文を使う場合は、別途`mt5_demo`またはMQL5 EAとして実装し、十分に検証

## 13. 重要な実装上の注意

このMVPのローカルDBは「この戦略が建てたポジション」の台帳です。手動売買や別のEAと同じ口座を混在させないでください。

`mt5_paper`は実注文を送りませんが、MT5端末から取得した価格データに依存します。MT5端末の未起動、未ログイン、銘柄未表示、通信断では実行に失敗します。

このリポジトリは、実口座ですぐ稼働させる完成品ではなく、戦略とリスク管理を検証するための出発点です。
