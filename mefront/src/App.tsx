import { useEffect, useReducer, useState } from "react";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { DropZone } from "./components/DropZone";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { ResultPanel } from "./components/ResultPanel";
import { useDetectionResults } from "./hooks/useDetectionResults";
import { useImageDetection } from "./hooks/useImageDetection";
import { appReducer, initialState } from "./lib/appState";
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

	const handleFileSelect = useImageDetection(dispatch, modelMetadata);

	const { displayDetections, exportDetections, isGeoTIFF, geoMeta, isMerged } =
		useDetectionResults(detectionSets, confidenceThreshold);

	const isProcessing =
		status.type === "loading" || status.type === "processing";
	const hasImage = currentImage != null;

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

			<div className="grid gap-6 lg:grid-cols-[1fr_300px]">
				<div className="space-y-4">
					{!hasImage && (
						<DropZone onFileSelect={handleFileSelect} disabled={isProcessing} />
					)}

					{hasImage && (
						<>
							<DetectionCanvas
								imageSource={currentImage.source}
								detections={displayDetections}
								imageWidth={currentImage.width}
								imageHeight={currentImage.height}
								onFileSelect={handleFileSelect}
								disabled={isProcessing}
							/>
							<div className="flex items-center gap-4">
								<button
									type="button"
									onClick={() => dispatch({ type: "CLEAR_ALL" })}
									disabled={isProcessing}
									className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
								>
									別の画像を選択
								</button>
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

// ---------------------------------------------------------------------------
// StatusMessage – renders the current status with an optional spinner.
// ---------------------------------------------------------------------------

function StatusMessage({
	status,
}: {
	status: import("./lib/appState").AppStatus;
}) {
	if (status.type === "idle") return null;

	const isActive = status.type === "loading" || status.type === "processing";

	return (
		<div className="flex items-center gap-2">
			{isActive && <LoadingSpinner size="sm" />}
			<span
				className={
					isActive
						? "text-sm font-medium text-gray-700 dark:text-gray-300"
						: "text-sm text-gray-500 dark:text-gray-400"
				}
			>
				{status.message}
			</span>
		</div>
	);
}

// ---------------------------------------------------------------------------
// MetadataUrlInput – lets the user specify a model metadata JSON URL.
// ---------------------------------------------------------------------------

function MetadataUrlInput({
	value,
	onChange,
	disabled,
}: {
	value: string;
	onChange: (url: string) => void;
	disabled: boolean;
}) {
	const [draft, setDraft] = useState(value);

	const commit = () => {
		const trimmed = draft.trim();
		if (trimmed && trimmed !== value) {
			onChange(trimmed);
		}
	};

	return (
		<div className="mb-4 flex flex-col gap-1">
			<label
				htmlFor="metadata-url"
				className="text-xs font-medium text-gray-600 dark:text-gray-400"
			>
				モデルメタデータ URL（JSON ファイル）
			</label>
			<div className="flex gap-2">
				<input
					id="metadata-url"
					type="url"
					value={draft}
					onChange={(e) => setDraft(e.target.value)}
					onBlur={commit}
					onKeyDown={(e) => {
						if (e.key === "Enter") {
							e.currentTarget.blur();
						}
					}}
					disabled={disabled}
					placeholder="https://example.com/model.json"
					className="min-w-0 flex-1 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-800 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:placeholder-gray-500"
				/>
				<button
					type="button"
					onClick={commit}
					disabled={disabled || draft.trim() === value}
					className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-blue-500 dark:hover:bg-blue-600"
				>
					適用
				</button>
			</div>
		</div>
	);
}

export default App;
