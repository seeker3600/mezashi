# mefront — AI agent instructions

## Project overview
Browser SPA for satellite/aerial image object-detection inference.  
Stack: **React 19 + TypeScript** / Bundler: **Vite** / Style: **Tailwind CSS v4** / Lint & Format: **Biome** / Tests: **Vitest + Testing Library**

## Tooling — prefer dedicated tools over shell

| Task | Preferred approach | Shell fallback |
|---|---|---|
| Run tests | VS Code *Run Tests* (Vitest) | `pnpm test` |
| Run single test | VS Code *Run Tests* with filter | `pnpm exec vitest run src/lib/__tests__/foo.test.ts` |
| Watch tests | — | `pnpm test:watch` |
| Install deps | — | `pnpm install` |
| Add dep | — | `pnpm add <pkg>` |
| Add dev dep | — | `pnpm add -D <pkg>` |
| Lint & format | VS Code Biome extension (format-on-save) | `pnpm exec biome check --write .` |
| Dev server | — | `pnpm dev` |
| Build | — | `pnpm build` (`tsc -b && vite build`) |

> When running from the monorepo root, use `pnpm -C mefront/ ...`.

## Test-first development
1. **Write a failing test first** in `src/lib/__tests__/<module>.test.ts` (or co-located `*.test.tsx` for components).
2. Implement the minimum code to make the test pass.
3. Refactor if needed, keeping tests green.

### Testing conventions
- Framework: **Vitest** (jsdom environment) + `@testing-library/react`.
- Use `describe` / `it` / `expect` from `vitest`.
- Use `import type` for type-only imports.
- Keep tests small and focused on one behaviour each.

```ts
// Example: src/lib/__tests__/labels.test.ts
import { describe, expect, it } from "vitest";
import { CONFIDENCE_THRESHOLD } from "../labels";

describe("labels", () => {
	it("CONFIDENCE_THRESHOLD should be a valid probability", () => {
		expect(CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
		expect(CONFIDENCE_THRESHOLD).toBeLessThan(1);
	});
});
```

## Coding rules
- Keep changes **minimal and focused**. Do not touch files unrelated to the task.
- No cosmetic-only edits (whitespace, blank lines, import reordering) in untouched code.
- Prefer **reasonable DRY** over perfect backward compatibility.
- All new code in **TypeScript** (`.ts` / `.tsx`). Do not add `.js` files.
- Use `import type { ... }` for type-only imports.
- Avoid `any`; if unavoidable, add a justifying comment.
- React components: function components + Hooks only.
- Tailwind for styling; minimize custom CSS.
- Package manager is **pnpm only** — never introduce npm / yarn / bun lockfiles.
- Lint & format is **Biome only** — do not add ESLint rules or Prettier config beyond what exists.
- ソースを修正した場合、最後にテストとlintを必ず通すこと。バグったまま完了しないでください。

## Language policy
- Issue / PR text → **日本語**
- Code, comments, commit messages → English (OK)

## AI agent best practices
- **Read before edit.** Always read the target file and surrounding context before proposing changes.
- **Scope check.** If the request touches model metadata or label maps, verify whether `medetect/` side also needs updating — but do NOT silently edit medetect files.
- **No speculation.** If unclear, search the codebase or ask rather than guessing API signatures.
- **No new files unless necessary.** Prefer editing existing files over creating new ones.
- **Verify after edit.** Run `pnpm exec biome check --write .` and the relevant tests after making changes.
- **One concern per commit.** Do not bundle unrelated fixes.
- **Environment variables.** Only `VITE_`-prefixed vars are exposed to the browser. Never embed secrets in frontend code.
