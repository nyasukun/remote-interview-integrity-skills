# Privacy and publication policy

このリポジトリは、再利用可能なスキル実装だけを公開します。
実案件のデータは、加工済みであっても公開対象にしません。

## Never commit

- 実案件の面談録画、静止画、音声、口元crop、画面共有
- 文字起こし、字幕、応募フォーム、履歴書、職務経歴書
- 氏名、表示名、メール、電話番号、住所、アカウントID
- 実案件の時刻一覧、event ID、媒体hash、Drive URL
- 分析結果、注釈、contact sheet、比較動画、QA出力
- API key、token、cookie、SSH/private key、`.env`

匿名化だけでは公開可としません。
実データ由来の値は、再同定や案件推測につながる可能性があるためです。

## Synthetic layout reference exception

動画rendererの既定レイアウトを固定するため、次の2ファイルだけを合成レイアウト参照として同梱します。

- `skills/interview-av-integrity/assets/layout-references/av-integrity-closure-review-approved.png`
- `skills/interview-voice-signal-comparison/assets/layout-references/voice-signal-designated-comparison-approved.png`

この2ファイルは実案件の録画、人物、音声、測定値から作成した証拠資料ではありません。
用途はrendererの視覚設計と品質検査に限り、分析結果として扱いません。

例外は、`scripts/audit_public_release.py`に固定したパス、SHA-256、1672×941 pxの寸法がすべて一致するときだけ成立します。
画像を変更する場合や別の媒体を追加する場合は、合成物であることを再確認し、privacy reviewを経て監査条件を明示的に更新します。
実案件由来の静止画を、この例外パスへ置き換えて公開することはできません。

## Before every push

1. `git status --short`と`git diff --cached --stat`で対象を確定する。
2. `python3 scripts/audit_public_release.py`でworking tree全体を検査する。
3. `python3 scripts/audit_public_release.py --staged`でcommit対象のindex blobとmodeを検査する。
4. staged file名に、許可した2つ以外の媒体、case、transcript、outputがないことを確認する。
5. `git diff --cached`を目視し、例が合成値と架空ラベルだけであることを確認する。
6. commit後、push前に`git show --stat --oneline HEAD`を再確認する。

`--staged`はworking tree上の同名ファイルではなく、Git indexに保存されたbyte列を検査します。
そのため、stage後にworking treeだけを修正しても、commit対象の検査結果は変わりません。

誤って機密をcommitした場合、単なる削除commitでは履歴から消えません。
pushを止め、credentialを失効し、履歴を安全に書き換えてから公開してください。
