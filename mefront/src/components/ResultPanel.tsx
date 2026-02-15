import {
	downloadGeoJSON,
	downloadMergedGeoJSON,
	downloadResultJSON,
} from "../lib/exportResults";
import type { Detection, GeoTIFFMeta } from "../lib/types";

interface ResultPanelProps {
	detections: Detection[];
	imageWidth: number;
	imageHeight: number;
	isGeoTIFF: boolean;
	geoMeta?: GeoTIFFMeta;
	confidenceThreshold: number;
	onConfidenceChange: (value: number) => void;
	isMerged?: boolean;
}

export function ResultPanel({
	detections,
	imageWidth,
	imageHeight,
	isGeoTIFF,
	geoMeta,
	confidenceThreshold,
	onConfidenceChange,
	isMerged = false,
}: ResultPanelProps) {
	// Group detections by class
	const classCounts = new Map<string, number>();
	for (const d of detections) {
		classCounts.set(d.className, (classCounts.get(d.className) ?? 0) + 1);
	}

	const handleDownload = () => {
		if (isGeoTIFF && geoMeta) {
			if (isMerged) {
				downloadMergedGeoJSON(detections, geoMeta);
			} else {
				downloadGeoJSON(detections, geoMeta);
			}
		} else {
			downloadResultJSON(detections, imageWidth, imageHeight);
		}
	};

	return (
		<div className="rounded-lg border border-gray-200 p-4 dark:border-gray-700">
			<h2 className="mb-3 text-lg font-semibold text-gray-800 dark:text-gray-200">
				検出結果
			</h2>

			<div className="mb-3 text-sm text-gray-600 dark:text-gray-400">
				<p>
					画像サイズ: {imageWidth} × {imageHeight}px
				</p>
				<p>検出数: {detections.length}</p>
				{isGeoTIFF && (
					<>
						<p className="text-blue-600 dark:text-blue-400">
							GeoTIFF (EPSG:{geoMeta?.epsg ?? "不明"})
						</p>
						{isMerged && (
							<p className="text-green-600 dark:text-green-400">
								統合済み (重複排除)
							</p>
						)}
					</>
				)}
			</div>

			<div className="mb-4">
				<label className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
					信頼度しきい値: {confidenceThreshold.toFixed(2)}
					<input
						type="range"
						min="0.05"
						max="0.95"
						step="0.05"
						value={confidenceThreshold}
						onChange={(e) => onConfidenceChange(Number(e.target.value))}
						className="mt-1 w-full"
					/>
				</label>
			</div>

			{classCounts.size > 0 && (
				<details className="mb-4">
					<summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300">
						クラス別検出数
					</summary>
					<ul className="mt-1 space-y-1 text-sm">
						{[...classCounts.entries()].map(([name, count]) => (
							<li
								key={name}
								className="flex justify-between text-gray-600 dark:text-gray-400"
							>
								<span>{name}</span>
								<span className="font-mono">{count}</span>
							</li>
						))}
					</ul>
				</details>
			)}

			<button
				type="button"
				onClick={handleDownload}
				disabled={detections.length === 0}
				className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{isGeoTIFF
					? isMerged
						? "統合GeoJSON をダウンロード"
						: "GeoJSON をダウンロード"
					: "JSON をダウンロード"}
			</button>
		</div>
	);
}
