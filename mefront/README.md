# mezashi フロントエンド

衛星・航空画像に対して、ブラウザ内で物体検出を実行する React アプリです。ONNX Runtime Web（WASM）で ONNX モデルを実行するため、画像を推論用サーバーへアップロードしません。

## 機能

- GeoTIFF (`.tif` / `.tiff`) と PNG、JPEG などのラスター画像を読み込み
- YOLO の検出モデルおよび OBB（回転矩形）モデルを実行
- 検出枠、ラベル、OBB の推定方向を画像に重ねて表示
- 信頼度しきい値、クラス別件数、検出結果を確認
- 通常画像は JSON、GeoTIFF は GeoJSON として結果をダウンロード
- モデルメタデータ JSON の URL を変更してモデルを切り替え
- 任意で入力画像拡張を有効化

GeoTIFF では画像に含まれる位置情報を結果に使用します。その他の画像では、モデルの既定値または入力された GSD（m/px）を使います。

## 必要環境

- Node.js 24.x LTS
- pnpm

## 起動

```bash
pnpm install
pnpm dev
```

起動後、ターミナルに表示される URL を開きます。ローカルネットワークから確認する場合は、Vite のオプションを使います。

```bash
pnpm dev -- --host
```

## 操作

1. モデルメタデータ JSON を確認し、必要なら URL を変更して「適用」を選択します。
2. 画像をドロップ、またはファイル選択で読み込みます。
3. 非 GeoTIFF 画像では GSD を入力します。
4. 推論後、表示する OBB 枠・ラベル・方向を切り替え、信頼度しきい値を調整します。
5. 結果パネルから JSON または GeoJSON をダウンロードします。

## モデルメタデータ

アプリは JSON メタデータから ONNX ファイルの URL、タスク種別、入力サイズ、ラベル、想定解像度、ライセンスを読み込みます。最低限、次のフィールドを指定します。

```json
{
	"name": "Example OBB model",
	"task": "obb",
	"onnxUrl": "https://example.com/model.onnx",
	"inputSize": 640,
	"labels": ["class-a", "class-b"],
	"expectedResolution": 1.0
}
```

`task` には `detect` または `obb` を指定します。`onnxUrl` はメタデータ JSON から解決可能な URL にしてください。公開するモデルには `license` フィールドを加え、利用条件を明示することを推奨します。

## コマンド

| コマンド | 説明 |
| --- | --- |
| `pnpm dev` | 開発サーバーを起動 |
| `pnpm build` | TypeScript を検査して本番用にビルド |
| `pnpm preview` | ビルド済み成果物をローカルで確認 |
| `pnpm test` | Vitest のテストを実行 |
| `pnpm test:watch` | テストをウォッチモードで実行 |
| `pnpm exec biome ci .` | Biome による整形・静的解析を実行 |
| `pnpm licenses` | サードパーティーライセンス一覧を生成 |

## 技術スタック

- React 19 / TypeScript
- Vite
- Tailwind CSS v4
- ONNX Runtime Web
- geotiff.js
- Vitest / Testing Library

## ライセンス

アプリの画面下部に、ビルド時に生成されるサードパーティーライセンス一覧へのリンクがあります。モデルと学習データセットの利用条件はそれぞれ異なるため、利用するモデルメタデータの `license` と、必要に応じて [DOTA の利用条件](https://captain-whu.github.io/DOTA/dataset.html) を確認してください。
