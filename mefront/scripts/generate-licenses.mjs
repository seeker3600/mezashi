#!/usr/bin/env node
/**
 * generate-licenses.mjs
 *
 * medetect (Python) と mefront (Node.js) の依存ライセンス情報を
 * 静的 HTML ページ (public/licenses.html) として生成する。
 *
 * Usage:
 *   node scripts/generate-licenses.mjs
 *
 * 前提:
 *   - medetect 側: `pixi run pip-licenses --format=json --with-urls`
 *     の出力を scripts/medetect-licenses.json に保存済み
 *   - mefront 側: npx license-checker --json --production を実行
 */

import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// ---------- オーバーライド ----------
/** @type {Record<string, {license?: string, url?: string}>} */
const OVERRIDES = JSON.parse(
	readFileSync(join(__dirname, "license-overrides.json"), "utf-8"),
);

/**
 * UNKNOWN またはローカルパスを除去し、http(s) URL のみ返す。
 * @param {string} raw
 */
function sanitizeUrl(raw) {
	if (!raw || raw === "UNKNOWN") return "";
	try {
		const u = new URL(raw);
		return u.protocol === "https:" || u.protocol === "http:" ? raw : "";
	} catch {
		return "";
	}
}

// ---------- medetect ----------
function loadMedetectLicenses() {
	const raw = readFileSync(join(__dirname, "medetect-licenses.json"), "utf-8");
	/** @type {Array<{Name: string, Version: string, License: string, URL: string}>} */
	const data = JSON.parse(raw);
	return data
		.filter((d) => d.Name !== "medetect" && d.Name !== "pip-licenses")
		.map((d) => {
			const ov = OVERRIDES[d.Name] ?? {};
			return {
				name: d.Name,
				version: d.Version,
				license:
					(d.License === "UNKNOWN" ? ov.license : d.License) ?? "UNKNOWN",
				url: sanitizeUrl(ov.url ?? d.URL ?? ""),
			};
		});
}

// ---------- mefront ----------
function loadMefrontLicenses() {
	const output = execSync("npx --yes license-checker --json --production", {
		cwd: ROOT,
		encoding: "utf-8",
		stdio: ["pipe", "pipe", "pipe"],
	});
	/** @type {Record<string, {licenses: string, repository?: string, publisher?: string}>} */
	const data = JSON.parse(output);
	return Object.entries(data)
		.filter(([key]) => !key.startsWith("mefront@"))
		.map(([key, val]) => {
			const match = key.match(/^(.+)@([^@]+)$/);
			const name = match ? match[1] : key;
			const ov = OVERRIDES[name] ?? {};
			return {
				name,
				version: match ? match[2] : "",
				license:
					(val.licenses === "Unknown" ? ov.license : val.licenses) ?? "Unknown",
				url: sanitizeUrl(ov.url ?? val.repository ?? ""),
			};
		});
}

// ---------- HTML 生成 ----------
/** @param {string} s */
function escapeHtml(s) {
	return s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

/**
 * @param {string} title
 * @param {Array<{name: string, version: string, license: string, url: string}>} deps
 */
function renderSection(title, deps) {
	const sorted = [...deps].sort((a, b) =>
		a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
	);
	const rows = sorted
		.map((d) => {
			const nameCell = d.url
				? `<a href="${escapeHtml(d.url)}" target="_blank" rel="noopener">${escapeHtml(d.name)}</a>`
				: escapeHtml(d.name);
			return `        <tr>
          <td>${nameCell}</td>
          <td>${escapeHtml(d.version)}</td>
          <td>${escapeHtml(d.license)}</td>
        </tr>`;
		})
		.join("\n");

	return `    <section>
      <h2>${escapeHtml(title)}</h2>
      <table>
        <thead>
          <tr><th>Package</th><th>Version</th><th>License</th></tr>
        </thead>
        <tbody>
${rows}
        </tbody>
      </table>
    </section>`;
}

function generateHtml(medetectDeps, mefrontDeps) {
	return `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Third-Party Licenses — mezashi</title>
  <style>
    :root { color-scheme: light dark; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 960px;
      margin: 0 auto;
      padding: 2rem 1rem;
      line-height: 1.6;
    }
    h1 { margin-bottom: 0.5rem; }
    h1 + p { margin-bottom: 2rem; color: #666; }
    h2 { margin: 2rem 0 1rem; padding-bottom: 0.25rem; border-bottom: 1px solid #ccc; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; }
    th { background: #f5f5f5; font-weight: 600; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (prefers-color-scheme: dark) {
      th { background: #1c1c1c; }
      td, th { border-bottom-color: #333; }
      h1 + p { color: #999; }
      h2 { border-bottom-color: #444; }
      a { color: #58a6ff; }
    }
    .back-link { display: inline-block; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <a class="back-link" href="./">&larr; Back to App</a>
  <main>
    <h1>Third-Party Licenses</h1>
    <p>This page lists the open-source licenses of third-party software used by the mezashi project.</p>
${renderSection("medetect – inference runtime (Python)", medetectDeps)}
${renderSection("mefront (Node.js / Browser)", mefrontDeps)}
  </main>
  <footer style="margin-top:3rem;padding-top:1rem;border-top:1px solid #ccc;color:#888;font-size:0.85rem;">
    <p>Generated on ${new Date().toISOString().split("T")[0]}. This project itself is licensed under
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" target="_blank" rel="noopener">AGPL-3.0</a>.</p>
  </footer>
</body>
</html>
`;
}

// ---------- main ----------
console.log("Collecting medetect licenses...");
const medetectDeps = loadMedetectLicenses();
console.log(`  Found ${medetectDeps.length} packages`);

console.log("Collecting mefront licenses...");
const mefrontDeps = loadMefrontLicenses();
console.log(`  Found ${mefrontDeps.length} packages`);

const html = generateHtml(medetectDeps, mefrontDeps);

const outDir = join(ROOT, "public");
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, "licenses.html");
writeFileSync(outPath, html, "utf-8");

console.log(`Wrote ${outPath}`);
