import { type Dispatch, useCallback } from "react";
import type { AppAction } from "../lib/appState";
import { imageDataToCanvas, isGeoTIFFFile, parseGeoTIFF } from "../lib/geotiff";
import { loadImageFromFile } from "../lib/imageUtils";
import { runInference } from "../lib/inference";
import type { ModelMetadata } from "../lib/types";

/**
 * Hook that returns a file-select handler.
 * It loads the image, runs inference, and dispatches the result into the app state.
 */
export function useImageDetection(
	dispatch: Dispatch<AppAction>,
	modelMetadata: ModelMetadata | null,
) {
	return useCallback(
		async (files: File[]) => {
			if (!modelMetadata) {
				dispatch({
					type: "SET_STATUS",
					status: {
						type: "error",
						message:
							"モデルメタデータが読み込まれていません。URL を確認してください。",
					},
				});
				return;
			}

			for (const file of files) {
				dispatch({
					type: "SET_STATUS",
					status: { type: "loading", message: "画像を読み込んでいます…" },
				});

				try {
					let src: HTMLCanvasElement | HTMLImageElement;
					let w: number;
					let h: number;
					let isGeoTIFF = false;
					let geoMeta: import("../lib/types").GeoTIFFMeta | undefined;

					if (isGeoTIFFFile(file)) {
						dispatch({
							type: "SET_STATUS",
							status: {
								type: "loading",
								message: "GeoTIFF を解析しています…",
							},
						});
						const result = await parseGeoTIFF(file);
						src = imageDataToCanvas(result.imageData);
						w = result.imageData.width;
						h = result.imageData.height;
						isGeoTIFF = true;
						geoMeta = result.meta;
					} else {
						const img = await loadImageFromFile(file);
						src = img;
						w = img.naturalWidth;
						h = img.naturalHeight;
					}

					dispatch({
						type: "SET_STATUS",
						status: {
							type: "processing",
							message: "推論中…",
							done: 0,
							total: 1,
						},
					});

					const detections = await runInference(
						src,
						w,
						h,
						modelMetadata,
						(done, total) => {
							dispatch({
								type: "SET_STATUS",
								status: {
									type: "processing",
									message: `推論中… (${done}/${total} タイル)`,
									done,
									total,
								},
							});
						},
					);

					dispatch({
						type: "ADD_RESULT",
						image: { source: src, width: w, height: h },
						detections,
						isGeoTIFF,
						geoMeta,
					});

					dispatch({
						type: "SET_STATUS",
						status: {
							type: "success",
							message: `検出完了: ${detections.length} 件`,
						},
					});
				} catch (err) {
					dispatch({
						type: "SET_STATUS",
						status: {
							type: "error",
							message: `エラー: ${err instanceof Error ? err.message : String(err)}`,
						},
					});
				}
			}
		},
		[dispatch, modelMetadata],
	);
}
