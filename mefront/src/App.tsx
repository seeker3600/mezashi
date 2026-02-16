import { useCallback, useMemo, useState } from "react";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { DropZone } from "./components/DropZone";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { ResultPanel } from "./components/ResultPanel";
import { mergeGeoTIFFDetections } from "./lib/exportResults";
import { imageDataToCanvas, isGeoTIFFFile, parseGeoTIFF } from "./lib/geotiff";
import { loadImageFromFile } from "./lib/imageUtils";
import { runInference } from "./lib/inference";
import { CONFIDENCE_THRESHOLD } from "./lib/labels";
import type { Detection, GeoTIFFMeta } from "./lib/types";

interface ImageData {
	source: HTMLCanvasElement | HTMLImageElement;
	width: number;
	height: number;
	isGeoTIFF: boolean;
	geoMeta?: GeoTIFFMeta;
	detections: Detection[];
}

function App() {
	const [firstImage, setFirstImage] = useState<ImageData | null>(null);
	const [secondImage, setSecondImage] = useState<ImageData | null>(null);
	const [status, setStatus] = useState<string>("");
	const [isProcessing, setIsProcessing] = useState(false);
	const [confidenceThreshold, setConfidenceThreshold] =
		useState(CONFIDENCE_THRESHOLD);
	const [isMerged, setIsMerged] = useState(false);

	// Compute the current detections to display
	const {
		detections,
		imageSource,
		imageWidth,
		imageHeight,
		isGeoTIFF,
		geoMeta,
	} = useMemo(() => {
		if (!firstImage && !secondImage) {
			return {
				detections: [],
				imageSource: null,
				imageWidth: 0,
				imageHeight: 0,
				isGeoTIFF: false,
				geoMeta: undefined,
			};
		}

		// If we have merged results (both are GeoTIFF)
		if (
			firstImage &&
			secondImage &&
			firstImage.isGeoTIFF &&
			secondImage.isGeoTIFF &&
			firstImage.geoMeta &&
			secondImage.geoMeta
		) {
			const mergedDetections = mergeGeoTIFFDetections(
				firstImage.detections,
				firstImage.geoMeta,
				secondImage.detections,
				secondImage.geoMeta,
			).filter((d) => d.confidence >= confidenceThreshold);

			return {
				detections: mergedDetections,
				imageSource: firstImage.source,
				imageWidth: firstImage.width,
				imageHeight: firstImage.height,
				isGeoTIFF: true,
				geoMeta: firstImage.geoMeta,
			};
		}

		// If we have a second image (non-merged case)
		if (secondImage) {
			return {
				detections: secondImage.detections.filter(
					(d) => d.confidence >= confidenceThreshold,
				),
				imageSource: secondImage.source,
				imageWidth: secondImage.width,
				imageHeight: secondImage.height,
				isGeoTIFF: secondImage.isGeoTIFF,
				geoMeta: secondImage.geoMeta,
			};
		}

		// Only first image
		if (firstImage) {
			return {
				detections: firstImage.detections.filter(
					(d) => d.confidence >= confidenceThreshold,
				),
				imageSource: firstImage.source,
				imageWidth: firstImage.width,
				imageHeight: firstImage.height,
				isGeoTIFF: firstImage.isGeoTIFF,
				geoMeta: firstImage.geoMeta,
			};
		}

		// Fallback (should never happen)
		return {
			detections: [],
			imageSource: null,
			imageWidth: 0,
			imageHeight: 0,
			isGeoTIFF: false,
			geoMeta: undefined,
		};
	}, [firstImage, secondImage, confidenceThreshold]);

	const handleFileSelect = useCallback(
		async (file: File) => {
			setIsProcessing(true);
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

				setStatus("モデルを読み込んでいます…");
				const dets = await runInference(src, w, h, (done, total) => {
					setStatus(`推論中… (${done}/${total} タイル)`);
				});

				const newImageData: ImageData = {
					source: src,
					width: w,
					height: h,
					isGeoTIFF: geo,
					geoMeta: meta,
					detections: dets,
				};

				// Determine if this is the first or second image
				if (!firstImage) {
					setFirstImage(newImageData);
					setSecondImage(null);
					setIsMerged(false);
					setStatus(`検出完了: ${dets.length} 件`);
				} else {
					setSecondImage(newImageData);

					// Check if we should merge
					if (firstImage.isGeoTIFF && geo && firstImage.geoMeta && meta) {
						const merged = mergeGeoTIFFDetections(
							firstImage.detections,
							firstImage.geoMeta,
							dets,
							meta,
						);
						setIsMerged(true);
						setStatus(
							`検出完了: ${dets.length} 件 (統合結果: ${merged.length} 件)`,
						);
					} else {
						setIsMerged(false);
						setStatus(
							`検出完了: ${dets.length} 件 (2枚目の画像に置き換えました)`,
						);
					}
				}
			} catch (err) {
				setStatus(
					`エラー: ${err instanceof Error ? err.message : String(err)}`,
				);
			} finally {
				setIsProcessing(false);
			}
		},
		[firstImage],
	);

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
					{!firstImage && (
						<DropZone onFileSelect={handleFileSelect} disabled={isProcessing} />
					)}

					{imageSource && (
						<>
							<DetectionCanvas
								imageSource={imageSource}
								detections={detections}
								imageWidth={imageWidth}
								imageHeight={imageHeight}
								onFileSelect={
									firstImage && !secondImage ? handleFileSelect : undefined
								}
								disabled={isProcessing}
							/>
							<div className="flex items-center gap-4">
								<button
									type="button"
									onClick={() => {
										setFirstImage(null);
										setSecondImage(null);
										setStatus("");
										setIsMerged(false);
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
								className={
									isProcessing
										? "text-sm font-medium text-gray-700 dark:text-gray-300"
										: "text-sm text-gray-500 dark:text-gray-400"
								}
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
						isMerged={isMerged}
					/>
				)}
			</div>
		</div>
	);
}

export default App;
