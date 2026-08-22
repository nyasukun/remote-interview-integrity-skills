# Privacy and publication policy

このリポジトリは、再利用可能なスキル実装だけを公開します。
実案件のデータは、加工済みであっても公開対象にしません。

## Never commit

- 面談録画、静止画、音声、口元crop、画面共有
- 文字起こし、字幕、応募フォーム、履歴書、職務経歴書
- 氏名、表示名、メール、電話番号、住所、アカウントID
- 実案件の時刻一覧、event ID、媒体hash、Drive URL
- 分析結果、注釈、contact sheet、比較動画、QA出力
- API key、token、cookie、SSH/private key、`.env`

匿名化だけでは公開可としません。
実データ由来の値は、再同定や案件推測につながる可能性があるためです。

## Before every push

1. `git status --short`と`git diff --cached --stat`で対象を確定する。
2. staged file名に媒体・case・transcript・outputがないことを確認する。
3. secret scannerと個人情報pattern scanをstaged内容へ実行する。
4. `git diff --cached`を目視し、例が合成値・架空ラベルだけであることを確認する。
5. commit後、push前に`git show --stat --oneline HEAD`を再確認する。

誤って機密をcommitした場合、単なる削除commitでは履歴から消えません。
pushを止め、credentialを失効し、履歴を安全に書き換えてから公開してください。
