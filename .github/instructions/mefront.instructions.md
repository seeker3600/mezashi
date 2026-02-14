---
applyTo: "mezashi/mefront/**"
---

# このプロジェクトについて
- 目的: シンプルな Web サイト / SPA を React + TypeScript で開発する。
- 技術スタック: Vite, React, TypeScript, Tailwind CSS, Biome
- 前提ランタイム: Node.js v24.x（LTS）+ pnpm（Windows では nvm-windows 利用想定）

# 重要な方針（必ず守る）
- パッケージマネージャは pnpm 固定（npm / yarn / bun の導入や lockfile 追加はしない）。
- コード整形・静的解析は Biome に一本化（ESLint / Prettier の追加・設定はしない）。
- スタイルは Tailwind CSS を基本とし、独自 CSS は最小限にする（必要がなければ増やさない）。
- 依存追加は最小限。追加する場合は理由を明確にし、軽量な選択肢を優先する。

# よく使うコマンド（pnpm）
## セットアップ
- 依存インストール: `pnpm install`

## 開発
- 開発サーバ起動: `pnpm dev`
  - うまくいかない場合は下記も可: `pnpm exec vite`

## ビルド / プレビュー
- 本番ビルド: `pnpm build`（内部的に `vite build`）
- ローカルプレビュー: `pnpm preview`（内部的に `vite preview`）
  - 注意: `pnpm preview` の前に `pnpm build` が必要
  - `vite preview` はローカル確認用途であり本番サーバ用途ではない

## フォーマット / Lint（Biome）
- 自動修正込みチェック: `pnpm exec biome check --write .`
- CI 用（自動修正なしで検証）: `pnpm exec biome ci .`

# コーディング規約（React + TypeScript）
- すべて TS/TSX で実装し、JS を増やさない。
- React コンポーネントは関数コンポーネント + Hooks を基本とする。
- props / state は型を明示し、any は避ける（必要なら理由をコメント）。
- 型のみ import は `import type { ... }` を優先。
- 可能な限り小さく分割し、再利用する UI は `src/components/` 配下へ。
- Tailwind の className は読みやすい順序でまとめ、極端に長くなる場合は分割（小さな部品化）を優先。
  - class の合成が頻出でつらい場合のみ、軽量なユーティリティ導入を検討する（導入前に既存依存を確認）。

# Tailwind CSS の扱い
- Tailwind は Vite プラグイン（@tailwindcss/vite）前提。
- グローバル CSS 側は `@import "tailwindcss";` を基本として維持し、不要な PostCSS 周りの追加はしない。

# 環境変数（Vite）
- `.env*` に置く公開してよい値のみをフロントに渡す。
- アプリ側で参照する env は `VITE_` プレフィックスを付け、`import.meta.env` 経由で参照する。
- 秘密情報（API キー等）をフロントに埋め込まない。必要ならバックエンド化を提案する。

# デバッグ（VS Code + ブラウザ）
- 基本はブラウザ DevTools（Console/Network/React DevTools）で確認。
- VS Code でブレークポイントを使う場合:
  - 先に `pnpm dev` で起動
  - VS Code の Browser Debug（例: Debug: Open Link または launch.json）で localhost を開き、TSX 側で停止できることを確認する

# 変更を出すときのチェックリスト
- 変更範囲を最小化し、不要なファイル生成（別ツールの設定・lockfile など）を避ける。
- `pnpm exec biome check --write .` を通してから提出する。
- UI 変更はスクリーンショットや簡単な再現手順を添える。
- 既存の動作（dev/build/preview）を壊さないことを優先し、必要なら README に手順を追記する。
