import { downloadGeoJSON, downloadResultJSON } from "../lib/exportResults";
import type { Detection, GeoTIFFMeta } from "../lib/types";

interface ResultPanelProps {
	detections: Detection[];
	imageWidth: number;
	imageHeight: number;
	isGeoTIFF: boolean;
	geoMeta?: GeoTIFFMeta;
}

export function ResultPanel({
	detections,
	imageWidth,
	imageHeight,
	isGeoTIFF,
	geoMeta,
}: ResultPanelProps) {
	// Group detections by class
	const classCounts = new Map<string, number>();
	for (const d of detections) {
		classCounts.set(d.className, (classCounts.get(d.className) ?? 0) + 1);
	}

	const handleDownload = () => {
		if (isGeoTIFF && geoMeta) {
			downloadGeoJSON(detections, geoMeta);
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
					<p className="text-blue-600 dark:text-blue-400">
						GeoTIFF (EPSG:{geoMeta?.epsg ?? "不明"})
					</p>
				)}
			</div>

			{classCounts.size > 0 && (
				<div className="mb-4">
					<h3 className="mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">
						クラス別検出数
					</h3>
					<ul className="space-y-1 text-sm">
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
				</div>
			)}

			<button
				type="button"
				onClick={handleDownload}
				disabled={detections.length === 0}
				className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{isGeoTIFF ? "GeoJSON をダウンロード" : "JSON をダウンロード"}
			</button>
		</div>
	);
}
