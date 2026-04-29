import { useCallback, useEffect, useReducer, useState } from "react";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { DropZone } from "./components/DropZone";
import { GSDDialog } from "./components/GSDInput";
import { MetadataUrlInput } from "./components/MetadataUrlInput";
import { ModelLicenseInfo } from "./components/ModelLicenseInfo";
import { ResultPanel } from "./components/ResultPanel";
import { StatusMessage } from "./components/StatusMessage";
import { useDetectionResults } from "./hooks/useDetectionResults";
import { useImageDetection } from "./hooks/useImageDetection";
import { appReducer, initialState } from "./lib/appState";
import { isGeoTIFFFile } from "./lib/geotiff";
import { fetchModelMetadata } from "./lib/modelMetadata";

function App() {
	const [state, dispatch] = useReducer(appReducer, initialState);
	const {
		currentImage,
		detectionSets,
		status,
		confidenceThreshold,
		metadataUrl,
		modelMetadata,
		userGSD,
		inputAugmentationEnabled,
	} = state;

	// Load model metadata whenever the URL changes
	useEffect(() => {
		dispatch({ type: "SET_MODEL_METADATA", metadata: null });
		dispatch({
			type: "SET_STATUS",
			status: {
				type: "loading",
				message: "モデルメタデータを読み込んでいます…",
			},
		});
		fetchModelMetadata(metadataUrl)
			.then((metadata) => {
				dispatch({ type: "SET_MODEL_METADATA", metadata });
				dispatch({
					type: "SET_USER_GSD",
					value: metadata.expectedResolution ?? null,
				});
				dispatch({ type: "SET_STATUS", status: { type: "idle" } });
			})
			.catch((err: unknown) => {
				dispatch({
					type: "SET_STATUS",
					status: {
						type: "error",
						message: `メタデータ読み込みエラー: ${
							err instanceof Error ? err.message : String(err)
						}`,
					},
				});
			});
	}, [metadataUrl]);

	const runDetection = useImageDetection(dispatch, modelMetadata);

	const [pendingFiles, setPendingFiles] = useState<File[] | null>(null);
	const [showBoxes, setShowBoxes] = useState(true);
	const [showLabels, setShowLabels] = useState(true);

	// Intercept file selection: show GSD dialog for non-GeoTIFF files
	const handleFileSelect = useCallback(
		(files: File[]) => {
			const hasNonGeoTIFF = files.some((f) => !isGeoTIFFFile(f));
			if (hasNonGeoTIFF) {
				setPendingFiles(files);
			} else {
				runDetection(files, undefined, inputAugmentationEnabled);
			}
		},
		[inputAugmentationEnabled, runDetection],
	);

	const handleGSDConfirm = useCallback(
		(gsd: number | null) => {
			dispatch({ type: "SET_USER_GSD", value: gsd });
			if (pendingFiles) {
				runDetection(pendingFiles, gsd ?? undefined, inputAugmentationEnabled);
			}
			setPendingFiles(null);
		},
		[inputAugmentationEnabled, pendingFiles, runDetection],
	);

	const handleGSDCancel = useCallback(() => {
		setPendingFiles(null);
	}, []);

	const { displayDetections, exportDetections, isGeoTIFF, geoMeta, isMerged } =
		useDetectionResults(detectionSets, confidenceThreshold);

	const isProcessing =
		status.type === "loading" || status.type === "processing";
	const hasImage = currentImage != null;
	const detectionCanvasKey = hasImage
		? `${detectionSets.length}:${currentImage.width}x${currentImage.height}`
		: "empty";

	return (
		<div className="mx-auto flex min-h-screen max-w-6xl flex-col p-4">
			<header className="mb-6">
				<h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
					物体検出
				</h1>
				<p className="text-sm text-gray-500 dark:text-gray-400">
					画像を読み込むと自動で物体検出を実行します
				</p>
			</header>

			<MetadataUrlInput
				value={metadataUrl}
				onChange={(url) => dispatch({ type: "SET_METADATA_URL", url })}
				disabled={isProcessing}
			/>

			<ModelLicenseInfo
				name={modelMetadata?.name}
				license={modelMetadata?.license}
			/>
			<div className="mb-4">
				<label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
					<input
						type="checkbox"
						checked={inputAugmentationEnabled}
						aria-describedby="input-augmentation-note"
						onChange={(e) =>
							dispatch({
								type: "SET_INPUT_AUGMENTATION",
								enabled: e.target.checked,
							})
						}
						disabled={isProcessing}
						className="h-4 w-4 accent-blue-600"
					/>
					入力画像拡張を有効化
					<span
						id="input-augmentation-note"
						className="text-xs text-gray-500 dark:text-gray-400"
					>
						(GaussianBlur / CLAHE / Brightness・Contrast / 上下左右反転)
					</span>
				</label>
			</div>

			{pendingFiles && (
				<GSDDialog
					modelDefault={modelMetadata?.expectedResolution}
					initialValue={userGSD}
					onConfirm={handleGSDConfirm}
					onCancel={handleGSDCancel}
				/>
			)}

			<div className="grid gap-6 lg:grid-cols-[1fr_300px]">
				<div className="space-y-4">
					{!hasImage && (
						<DropZone onFileSelect={handleFileSelect} disabled={isProcessing} />
					)}

					{hasImage && (
						<>
							<DetectionCanvas
								key={detectionCanvasKey}
								imageSource={currentImage.source}
								detections={displayDetections}
								imageWidth={currentImage.width}
								imageHeight={currentImage.height}
								onFileSelect={handleFileSelect}
								disabled={isProcessing}
								showBoxes={showBoxes}
								showLabels={showLabels}
							/>
							<div className="flex items-center gap-4">
								<button
									type="button"
									onClick={() => dispatch({ type: "CLEAR_ALL" })}
									disabled={isProcessing}
									className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
								>
									別の画像を選択
								</button>{" "}
								<label className="flex items-center gap-1.5 text-sm text-gray-700 dark:text-gray-300">
									<input
										type="checkbox"
										checked={showBoxes}
										onChange={(e) => setShowBoxes(e.target.checked)}
										className="h-4 w-4 accent-blue-600"
									/>
									OBB枠
								</label>
								<label className="flex items-center gap-1.5 text-sm text-gray-700 dark:text-gray-300">
									<input
										type="checkbox"
										checked={showLabels}
										onChange={(e) => setShowLabels(e.target.checked)}
										className="h-4 w-4 accent-blue-600"
									/>
									ラベル
								</label>{" "}
								<StatusMessage status={status} />
							</div>
						</>
					)}

					{!hasImage && status.type !== "idle" && (
						<StatusMessage status={status} />
					)}
				</div>

				{hasImage && (
					<ResultPanel
						detections={exportDetections}
						imageWidth={currentImage.width}
						imageHeight={currentImage.height}
						isGeoTIFF={isGeoTIFF}
						geoMeta={geoMeta}
						confidenceThreshold={confidenceThreshold}
						onConfidenceChange={(v) =>
							dispatch({ type: "SET_CONFIDENCE", value: v })
						}
						isMerged={isMerged}
					/>
				)}
			</div>

			<footer className="mt-auto pt-8 text-center text-xs text-gray-400 dark:text-gray-500">
				<a
					href="./licenses.html"
					className="underline hover:text-gray-600 dark:hover:text-gray-300"
				>
					Third-Party Licenses
				</a>
			</footer>
		</div>
	);
}

export default App;
