import { useCallback, useMemo, useState } from "react";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { DropZone } from "./components/DropZone";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { ResultPanel } from "./components/ResultPanel";
import { imageDataToCanvas, isGeoTIFFFile, parseGeoTIFF } from "./lib/geotiff";
import { loadImageFromFile } from "./lib/imageUtils";
import { runInference } from "./lib/inference";
import { CONFIDENCE_THRESHOLD } from "./lib/labels";
import type { Detection, GeoTIFFMeta } from "./lib/types";

function App() {
	const [imageSource, setImageSource] = useState<
		HTMLCanvasElement | HTMLImageElement | null
	>(null);
	const [imageWidth, setImageWidth] = useState(0);
	const [imageHeight, setImageHeight] = useState(0);
	const [rawDetections, setRawDetections] = useState<Detection[]>([]);
	const [isGeoTIFF, setIsGeoTIFF] = useState(false);
	const [geoMeta, setGeoMeta] = useState<GeoTIFFMeta | undefined>();
	const [status, setStatus] = useState<string>("");
	const [isProcessing, setIsProcessing] = useState(false);
	const [confidenceThreshold, setConfidenceThreshold] =
		useState(CONFIDENCE_THRESHOLD);

	const detections = useMemo(
		() => rawDetections.filter((d) => d.confidence >= confidenceThreshold),
		[rawDetections, confidenceThreshold],
	);

	const handleFileSelect = useCallback(async (file: File) => {
		setIsProcessing(true);
		setRawDetections([]);
		setStatus("画像を読み込んでいます…");

		try {
			let src: HTMLCanvasElement | HTMLImageElement;
			let w: number;
			let h: number;
			let geo = false;
			let meta: GeoTIFFMeta | undefined;

			if (isGeoTIFFFile(file)) {
				setStatus("GeoTIFF を解析しています…");
				const result = await parseGeoTIFF(file);
				const canvas = imageDataToCanvas(result.imageData);
				src = canvas;
				w = result.imageData.width;
				h = result.imageData.height;
				geo = true;
				meta = result.meta;
			} else {
				const img = await loadImageFromFile(file);
				src = img;
				w = img.naturalWidth;
				h = img.naturalHeight;
			}

			setImageSource(src);
			setImageWidth(w);
			setImageHeight(h);
			setIsGeoTIFF(geo);
			setGeoMeta(meta);

			setStatus("モデルを読み込んでいます…");
			const dets = await runInference(src, w, h, (done, total) => {
				setStatus(`推論中… (${done}/${total} タイル)`);
			});

			setRawDetections(dets);
			setStatus(`検出完了: ${dets.length} 件`);
		} catch (err) {
			setStatus(`エラー: ${err instanceof Error ? err.message : String(err)}`);
		} finally {
			setIsProcessing(false);
		}
	}, []);

	return (
		<div className="mx-auto min-h-screen max-w-6xl p-4">
			<header className="mb-6">
				<h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
					物体検出
				</h1>
				<p className="text-sm text-gray-500 dark:text-gray-400">
					画像を読み込むと自動で物体検出を実行します
				</p>
			</header>

			<div className="grid gap-6 lg:grid-cols-[1fr_300px]">
				<div className="space-y-4">
					{!imageSource && (
						<DropZone onFileSelect={handleFileSelect} disabled={isProcessing} />
					)}

					{imageSource && (
						<>
							<DetectionCanvas
								imageSource={imageSource}
								detections={detections}
								imageWidth={imageWidth}
								imageHeight={imageHeight}
							/>
							<div className="flex items-center gap-4">
								<button
									type="button"
									onClick={() => {
										setImageSource(null);
										setRawDetections([]);
										setStatus("");
										setIsGeoTIFF(false);
										setGeoMeta(undefined);
									}}
									disabled={isProcessing}
									className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
								>
									別の画像を選択
								</button>
								{isProcessing && (
									<div className="flex items-center gap-2">
										<LoadingSpinner size="sm" />
										{status && (
											<span className="text-sm font-medium text-gray-700 dark:text-gray-300">
												{status}
											</span>
										)}
									</div>
								)}
								{!isProcessing && status && (
									<span className="text-sm text-gray-500 dark:text-gray-400">
										{status}
									</span>
								)}
							</div>
						</>
					)}

					{!imageSource && status && (
						<div className="flex items-center gap-2">
							{isProcessing && <LoadingSpinner size="sm" />}
							<p
								className={`text-sm ${isProcessing ? "font-medium text-gray-700 dark:text-gray-300" : "text-gray-500 dark:text-gray-400"}`}
							>
								{status}
							</p>
						</div>
					)}
				</div>

				{imageSource && (
					<ResultPanel
						detections={detections}
						imageWidth={imageWidth}
						imageHeight={imageHeight}
						isGeoTIFF={isGeoTIFF}
						geoMeta={geoMeta}
						confidenceThreshold={confidenceThreshold}
						onConfidenceChange={setConfidenceThreshold}
					/>
				)}
			</div>
		</div>
	);
}

export default App;
