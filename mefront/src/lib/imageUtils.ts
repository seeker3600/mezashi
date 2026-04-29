export type TileAugmentation =
	| "none"
	| "gaussianBlur"
	| "clahe"
	| "brightnessContrast"
	| "flipHorizontal"
	| "flipVertical"
	| "flipBoth";

const ENABLED_TILE_AUGMENTATIONS: readonly TileAugmentation[] = [
	"none",
	"gaussianBlur",
	"clahe",
	"brightnessContrast",
	"flipHorizontal",
	"flipVertical",
	"flipBoth",
];

const BRIGHTNESS_CONTRAST_ALPHA = 1.15;
const BRIGHTNESS_CONTRAST_BETA = 12;
const CLAHE_TILE_SIZE = 8;
const CLAHE_HISTOGRAM_BINS = 256;
const CLAHE_CLIP_FACTOR = 2;

function clamp8(v: number): number {
	return Math.max(0, Math.min(255, Math.round(v)));
}

function flipCanvas(
	canvas: HTMLCanvasElement,
	flipX: boolean,
	flipY: boolean,
): void {
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	const temp = document.createElement("canvas");
	temp.width = canvas.width;
	temp.height = canvas.height;
	const tempCtx = temp.getContext("2d");
	if (!tempCtx) throw new Error("Cannot get 2d context");
	tempCtx.drawImage(canvas, 0, 0);

	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.save();
	ctx.translate(flipX ? canvas.width : 0, flipY ? canvas.height : 0);
	ctx.scale(flipX ? -1 : 1, flipY ? -1 : 1);
	ctx.drawImage(temp, 0, 0);
	ctx.restore();
}

function applyGaussianBlur(canvas: HTMLCanvasElement): void {
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	const { width, height } = canvas;
	const imageData = ctx.getImageData(0, 0, width, height);
	const src = imageData.data;
	const out = new Uint8ClampedArray(src.length);
	const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
	const sum = 16;

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			for (let c = 0; c < 3; c++) {
				let acc = 0;
				let k = 0;
				for (let ky = -1; ky <= 1; ky++) {
					const py = Math.max(0, Math.min(height - 1, y + ky));
					for (let kx = -1; kx <= 1; kx++) {
						const px = Math.max(0, Math.min(width - 1, x + kx));
						const idx = (py * width + px) * 4 + c;
						acc += src[idx] * kernel[k++];
					}
				}
				out[(y * width + x) * 4 + c] = clamp8(acc / sum);
			}
			out[(y * width + x) * 4 + 3] = src[(y * width + x) * 4 + 3];
		}
	}

	ctx.putImageData(new ImageData(out, width, height), 0, 0);
}

function applyBrightnessContrast(canvas: HTMLCanvasElement): void {
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	const { data } = imageData;

	for (let i = 0; i < data.length; i += 4) {
		data[i] = clamp8(
			(data[i] - 128) * BRIGHTNESS_CONTRAST_ALPHA +
				128 +
				BRIGHTNESS_CONTRAST_BETA,
		);
		data[i + 1] = clamp8(
			(data[i + 1] - 128) * BRIGHTNESS_CONTRAST_ALPHA +
				128 +
				BRIGHTNESS_CONTRAST_BETA,
		);
		data[i + 2] = clamp8(
			(data[i + 2] - 128) * BRIGHTNESS_CONTRAST_ALPHA +
				128 +
				BRIGHTNESS_CONTRAST_BETA,
		);
	}
	ctx.putImageData(imageData, 0, 0);
}

function applyClahe(canvas: HTMLCanvasElement): void {
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");
	const { width, height } = canvas;
	const imageData = ctx.getImageData(0, 0, width, height);
	const { data } = imageData;

	for (let ty = 0; ty < height; ty += CLAHE_TILE_SIZE) {
		for (let tx = 0; tx < width; tx += CLAHE_TILE_SIZE) {
			const tw = Math.min(CLAHE_TILE_SIZE, width - tx);
			const th = Math.min(CLAHE_TILE_SIZE, height - ty);
			const tilePixels = tw * th;
			const hist = new Array<number>(CLAHE_HISTOGRAM_BINS).fill(0);

			for (let y = ty; y < ty + th; y++) {
				for (let x = tx; x < tx + tw; x++) {
					const i = (y * width + x) * 4;
					const luma = Math.round(
						0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2],
					);
					hist[luma]++;
				}
			}

			const clipLimit = Math.max(
				1,
				Math.floor((CLAHE_CLIP_FACTOR * tilePixels) / CLAHE_HISTOGRAM_BINS),
			);
			let excess = 0;
			for (let i = 0; i < CLAHE_HISTOGRAM_BINS; i++) {
				if (hist[i] > clipLimit) {
					excess += hist[i] - clipLimit;
					hist[i] = clipLimit;
				}
			}
			const increment = Math.floor(excess / CLAHE_HISTOGRAM_BINS);
			const remainder = excess % CLAHE_HISTOGRAM_BINS;
			for (let i = 0; i < CLAHE_HISTOGRAM_BINS; i++) {
				hist[i] += increment + (i < remainder ? 1 : 0);
			}

			const cdf = new Array<number>(CLAHE_HISTOGRAM_BINS).fill(0);
			cdf[0] = hist[0];
			for (let i = 1; i < CLAHE_HISTOGRAM_BINS; i++) {
				cdf[i] = cdf[i - 1] + hist[i];
			}
			const cdfMin = cdf.find((v) => v > 0) ?? 0;
			const denom = Math.max(1, tilePixels - cdfMin);

			for (let y = ty; y < ty + th; y++) {
				for (let x = tx; x < tx + tw; x++) {
					const i = (y * width + x) * 4;
					const luma = Math.round(
						0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2],
					);
					const eq = clamp8(((cdf[luma] - cdfMin) / denom) * 255);
					const ratio = luma > 0 ? eq / luma : 0;
					data[i] = clamp8(data[i] * ratio);
					data[i + 1] = clamp8(data[i + 1] * ratio);
					data[i + 2] = clamp8(data[i + 2] * ratio);
				}
			}
		}
	}

	ctx.putImageData(imageData, 0, 0);
}

function applyTileAugmentation(
	canvas: HTMLCanvasElement,
	augmentation: TileAugmentation,
): void {
	switch (augmentation) {
		case "none":
			return;
		case "gaussianBlur":
			applyGaussianBlur(canvas);
			return;
		case "clahe":
			applyClahe(canvas);
			return;
		case "brightnessContrast":
			applyBrightnessContrast(canvas);
			return;
		case "flipHorizontal":
			flipCanvas(canvas, true, false);
			return;
		case "flipVertical":
			flipCanvas(canvas, false, true);
			return;
		case "flipBoth":
			flipCanvas(canvas, true, true);
			return;
	}
}

export function getTileAugmentations(
	enabled: boolean,
): readonly TileAugmentation[] {
	return enabled ? ENABLED_TILE_AUGMENTATIONS : ["none"];
}

/**
 * Load an image file (jpg/png) into an HTMLImageElement.
 */
export function loadImageFromFile(file: File): Promise<HTMLImageElement> {
	return new Promise((resolve, reject) => {
		const url = URL.createObjectURL(file);
		const img = new Image();
		img.onload = () => {
			URL.revokeObjectURL(url);
			resolve(img);
		};
		img.onerror = () => {
			URL.revokeObjectURL(url);
			reject(new Error("Failed to load image"));
		};
		img.src = url;
	});
}

/**
 * Extract pixel data from a canvas as Float32Array in CHW format (RGB, 0-1 normalized).
 * The canvas should already be sized to the desired dimensions.
 */
export function canvasToFloat32CHW(
	canvas: HTMLCanvasElement | OffscreenCanvas,
): Float32Array {
	const ctx =
		canvas instanceof HTMLCanvasElement
			? canvas.getContext("2d")
			: canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	const { width, height } = canvas;
	const imageData = ctx.getImageData(0, 0, width, height);
	const { data } = imageData;
	const chw = new Float32Array(3 * width * height);
	const planeSize = width * height;

	for (let i = 0; i < planeSize; i++) {
		chw[i] = data[i * 4] / 255; // R
		chw[planeSize + i] = data[i * 4 + 1] / 255; // G
		chw[2 * planeSize + i] = data[i * 4 + 2] / 255; // B
	}
	return chw;
}

/**
 * 画像の GSD をモデルの訓練解像度 (targetGSDMeters) に合わせるスケール係数を計算する。
 *
 * - GSD がモデル期待値より小さい（高解像度）→ 1 未満の縮小率
 * - GSD がモデル期待値より大きい（低解像度）→ 1 超の拡大率
 * - GSD が一致する場合 → 1.0
 *
 * @param pixelScale      画像の GSD (メートル/px)。投影座標系を前提とする。
 * @param targetGSDMeters モデルの訓練 GSD (メートル/px)。ModelMetadata.expectedResolution。
 * @returns scale > 0 のスケール係数
 */
export function computeGeoTIFFShrinkScale(
	pixelScale: { x: number; y: number },
	targetGSDMeters: number,
): number {
	const currentGSD = Math.max(pixelScale.x, pixelScale.y);
	return currentGSD / targetGSDMeters;
}

/**
 * 画像を指定の縮小率で縮小した Canvas を返す。
 */
export function createShrunkCanvas(
	src: HTMLCanvasElement | HTMLImageElement,
	srcWidth: number,
	srcHeight: number,
	scale: number,
): HTMLCanvasElement {
	const newWidth = Math.round(srcWidth * scale);
	const newHeight = Math.round(srcHeight * scale);
	const canvas = document.createElement("canvas");
	canvas.width = newWidth;
	canvas.height = newHeight;
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");
	ctx.drawImage(src, 0, 0, srcWidth, srcHeight, 0, 0, newWidth, newHeight);
	return canvas;
}

/**
 * Prepare a tile from the source image/canvas for the model input.
 * Draws the region (sx, sy, sw, sh) from src onto a MODEL_INPUT_SIZE canvas with letterboxing.
 * Returns the Float32Array in CHW format and the scale/padding info for coordinate mapping.
 */
export function prepareTile(
	src: HTMLCanvasElement | HTMLImageElement,
	sx: number,
	sy: number,
	sw: number,
	sh: number,
	modelSize: number,
	augmentation: TileAugmentation = "none",
): {
	input: Float32Array;
	scale: number;
	padX: number;
	padY: number;
} {
	const canvas = document.createElement("canvas");
	canvas.width = modelSize;
	canvas.height = modelSize;
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	// Letterbox: fit the tile into modelSize x modelSize
	const scale = Math.min(modelSize / sw, modelSize / sh);
	const newW = Math.round(sw * scale);
	const newH = Math.round(sh * scale);
	const padX = (modelSize - newW) / 2;
	const padY = (modelSize - newH) / 2;

	// Fill with gray (114/255 is YOLO's default padding)
	ctx.fillStyle = `rgb(114, 114, 114)`;
	ctx.fillRect(0, 0, modelSize, modelSize);
	ctx.drawImage(src, sx, sy, sw, sh, padX, padY, newW, newH);
	applyTileAugmentation(canvas, augmentation);

	return {
		input: canvasToFloat32CHW(canvas),
		scale,
		padX,
		padY,
	};
}
