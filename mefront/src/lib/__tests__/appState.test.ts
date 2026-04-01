import { describe, expect, it } from "vitest";
import { type AppState, appReducer, initialState } from "../appState";

describe("appReducer", () => {
	const dummyImage = {
		source: {} as HTMLCanvasElement,
		width: 800,
		height: 600,
	};

	it("should return initial state", () => {
		expect(initialState.currentImage).toBeNull();
		expect(initialState.detectionSets).toHaveLength(0);
		expect(initialState.status.type).toBe("idle");
	});

	describe("ADD_RESULT", () => {
		it("should set currentImage and append a detection set", () => {
			const dets = [
				{
					classId: 0,
					className: "plane",
					confidence: 0.9,
					cx: 10,
					cy: 20,
					width: 30,
					height: 40,
					angle: 0,
				},
			];
			const next = appReducer(initialState, {
				type: "ADD_RESULT",
				image: dummyImage,
				detections: dets,
				task: "obb",
				isGeoTIFF: false,
			});
			expect(next.currentImage).toBe(dummyImage);
			expect(next.detectionSets).toHaveLength(1);
			expect(next.detectionSets[0].detections).toBe(dets);
			expect(next.detectionSets[0].task).toBe("obb");
			expect(next.detectionSets[0].isGeoTIFF).toBe(false);
		});

		it("should accumulate multiple detection sets while replacing image", () => {
			const img1 = { ...dummyImage, width: 100 };
			const img2 = { ...dummyImage, width: 200 };
			let state = appReducer(initialState, {
				type: "ADD_RESULT",
				image: img1,
				detections: [],
				task: "obb",
				isGeoTIFF: true,
				geoMeta: {
					tiePoint: { x: 0, y: 0 },
					pixelScale: { x: 1, y: 1 },
					epsg: null,
				},
			});
			state = appReducer(state, {
				type: "ADD_RESULT",
				image: img2,
				detections: [],
				task: "obb",
				isGeoTIFF: true,
				geoMeta: {
					tiePoint: { x: 50, y: 0 },
					pixelScale: { x: 1, y: 1 },
					epsg: null,
				},
			});
			expect(state.currentImage).toBe(img2);
			expect(state.detectionSets).toHaveLength(2);
		});
	});

	describe("CLEAR_ALL", () => {
		it("should reset to initial state but preserve confidenceThreshold, metadataUrl, and modelMetadata", () => {
			const modified: AppState = {
				currentImage: dummyImage,
				detectionSets: [
					{ detections: [], task: "obb" as const, isGeoTIFF: false },
				],
				status: { type: "success", message: "done" },
				confidenceThreshold: 0.5,
				metadataUrl: "https://example.com/model.json",
				modelMetadata: null,
			};
			const next = appReducer(modified, { type: "CLEAR_ALL" });
			expect(next.currentImage).toBeNull();
			expect(next.detectionSets).toHaveLength(0);
			expect(next.status.type).toBe("idle");
			expect(next.confidenceThreshold).toBe(0.5);
			expect(next.metadataUrl).toBe("https://example.com/model.json");
		});
	});

	describe("SET_STATUS", () => {
		it("should update status", () => {
			const next = appReducer(initialState, {
				type: "SET_STATUS",
				status: { type: "loading", message: "loading..." },
			});
			expect(next.status).toEqual({ type: "loading", message: "loading..." });
		});
	});

	describe("SET_CONFIDENCE", () => {
		it("should update confidenceThreshold", () => {
			const next = appReducer(initialState, {
				type: "SET_CONFIDENCE",
				value: 0.75,
			});
			expect(next.confidenceThreshold).toBe(0.75);
		});
	});
});
