# 航空画像 物体検出 Web アプリ

航空・衛星画像に対して、ブラウザ完結で物体検出を実行する React アプリです。  
YOLO OBB (Oriented Bounding Box) モデルを ONNX Runtime Web (WASM) で動かします。

## 検出クラス

[DOTA データセット](https://captain-whu.github.io/DOTA/) の 15 クラスに対応しています。

`plane` / `ship` / `storage tank` / `baseball diamond` / `tennis court` / `basketball court` / `ground track field` / `harbor` / `bridge` / `large vehicle` / `small vehicle` / `helicopter` / `roundabout` / `soccer ball field` / `swimming pool`

## 対応画像フォーマット

- GeoTIFF (`.tif` / `.tiff`)
- 一般的なラスター画像 (`.png` / `.jpg` など)

## セットアップ

```bash
pnpm install
pnpm dev
```

## スクリプト

| コマンド | 説明 |
|---|---|
| `pnpm dev` | 開発サーバー起動 |
| `pnpm build` | プロダクションビルド |
| `pnpm preview` | ビルド結果のプレビュー |
| `pnpm test` | テスト実行 |

## 技術スタック

- React 19 / TypeScript
- Vite
- Tailwind CSS v4
- ONNX Runtime Web
- geotiff.js
