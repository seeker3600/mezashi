import { useCallback, useEffect, useRef, useState } from "react";
import type { DirectionDisplayMode } from "../lib/directionDisplay";
import {
	aggregateDirectionsByGrid,
	getDetailScaleThreshold,
	getDirectionDisplayMode,
} from "../lib/directionDisplay";
import {
	DIRECTION_MARKER_STYLE,
	DIRECTION_MARKER_STYLES,
	DIRECTION_OVERVIEW_GRID_SPACING,
} from "../lib/labels";
import { getOBBCorners, getOBBLongAxisAngle } from "../lib/obbUtils";
import type { Detection } from "../lib/types";

/** Convert hex color to rgba string */
function hexToRgba(hex: string, alpha: number): string {
	const r = Number.parseInt(hex.slice(1, 3), 16);
	const g = Number.parseInt(hex.slice(3, 5), 16);
	const b = Number.parseInt(hex.slice(5, 7), 16);
	return `rgba(${r},${g},${b},${alpha})`;
}

/** Palette for different class colors */
const COLORS = [
	"#FF3838",
	"#FF9D97",
	"#FF701F",
	"#FFB21D",
	"#CFD231",
	"#48F90A",
	"#92CC17",
	"#3DDB86",
	"#1A9334",
	"#00D4BB",
	"#2C99A8",
	"#00C2FF",
	"#344593",
	"#6473FF",
	"#0018EC",
];

const OVERVIEW_DIRECTION_COLOR = "#0F766E";

interface DirectionMarkerOptions {
	cx: number;
	cy: number;
	angle: number;
	length: number;
	lineWidth: number;
	color: string;
	alpha: number;
}

function drawDirectionMarker(
	ctx: CanvasRenderingContext2D,
	{ cx, cy, angle, length, lineWidth, color, alpha }: DirectionMarkerOptions,
): void {
	if (length <= 0) return;

	const directionX = Math.cos(angle);
	const directionY = Math.sin(angle);
	ctx.strokeStyle = hexToRgba(color, alpha);
	ctx.lineWidth = lineWidth;

	if (DIRECTION_MARKER_STYLE === DIRECTION_MARKER_STYLES.LINE) {
		ctx.beginPath();
		ctx.moveTo(cx - (directionX * length) / 2, cy - (directionY * length) / 2);
		ctx.lineTo(cx + (directionX * length) / 2, cy + (directionY * length) / 2);
		ctx.stroke();
		return;
	}

	const endX = cx + directionX * length;
	const endY = cy + directionY * length;
	const arrowHeadLength = Math.min(lineWidth * 4, length * 0.35);
	ctx.beginPath();
	ctx.moveTo(cx, cy);
	ctx.lineTo(endX, endY);
	ctx.stroke();

	if (arrowHeadLength <= 0) return;

	ctx.beginPath();
	ctx.moveTo(endX, endY);
	ctx.lineTo(
		endX - directionX * arrowHeadLength - directionY * arrowHeadLength * 0.6,
		endY - directionY * arrowHeadLength + directionX * arrowHeadLength * 0.6,
	);
	ctx.lineTo(
		endX - directionX * arrowHeadLength + directionY * arrowHeadLength * 0.6,
		endY - directionY * arrowHeadLength - directionX * arrowHeadLength * 0.6,
	);
	ctx.closePath();
	ctx.fillStyle = hexToRgba(color, alpha);
	ctx.fill();
}

interface DetectionCanvasProps {
	imageSource: HTMLCanvasElement | HTMLImageElement | null;
	detections: Detection[];
	imageWidth: number;
	imageHeight: number;
	onFileSelect?: (files: File[]) => void;
	disabled?: boolean;
	showBoxes?: boolean;
	showLabels?: boolean;
	showDirection?: boolean;
}

export function DetectionCanvas({
	imageSource,
	detections,
	imageWidth,
	imageHeight,
	onFileSelect,
	disabled = false,
	showBoxes = true,
	showLabels = true,
	showDirection = false,
}: DetectionCanvasProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [isDragOver, setIsDragOver] = useState(false);
	const [scale, setScale] = useState(1);
	const [offset, setOffset] = useState({ x: 0, y: 0 });
	const [isPanning, setIsPanning] = useState(false);
	const [startPan, setStartPan] = useState({ x: 0, y: 0 });
	const directionModeRef = useRef<DirectionDisplayMode>("overview");

	const handleDragOver = useCallback(
		(e: React.DragEvent) => {
			e.preventDefault();
			if (!disabled && onFileSelect) {
				setIsDragOver(true);
			}
		},
		[disabled, onFileSelect],
	);

	const handleDragLeave = useCallback(() => {
		setIsDragOver(false);
	}, []);

	const handleDrop = useCallback(
		(e: React.DragEvent) => {
			e.preventDefault();
			setIsDragOver(false);
			if (disabled || !onFileSelect) return;
			const files = Array.from(e.dataTransfer.files);
			if (files.length > 0) onFileSelect(files);
		},
		[disabled, onFileSelect],
	);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (disabled || !onFileSelect) return;
			if (e.key === "Enter" || e.key === " ") {
				e.preventDefault();
				// Trigger file input dialog
				const input = document.createElement("input");
				input.type = "file";
				input.accept = "image/*,.tif,.tiff";
				input.multiple = true;
				input.onchange = () => {
					const files = Array.from(input.files ?? []);
					if (files.length > 0) onFileSelect(files);
				};
				input.click();
			}
		},
		[disabled, onFileSelect],
	);

	const handleWheel = useCallback(
		(e: WheelEvent) => {
			e.preventDefault();
			const canvas = canvasRef.current;
			if (!canvas) return;

			const rect = canvas.getBoundingClientRect();
			const mouseX = e.clientX - rect.left;
			const mouseY = e.clientY - rect.top;

			// Calculate base offset (image centering)
			const scaleX = canvas.width / imageWidth;
			const scaleY = canvas.height / imageHeight;
			const fitScale = Math.min(scaleX, scaleY);
			const displayWidth = imageWidth * fitScale;
			const displayHeight = imageHeight * fitScale;
			const baseOffsetX = (canvas.width - displayWidth) / 2;
			const baseOffsetY = (canvas.height - displayHeight) / 2;

			const delta = e.deltaY > 0 ? 0.9 : 1.1;
			const newScale = Math.min(Math.max(1.0, scale * delta), 10);

			// Calculate world coordinates at mouse position
			const worldX = (mouseX - baseOffsetX - offset.x) / scale;
			const worldY = (mouseY - baseOffsetY - offset.y) / scale;

			// Adjust offset so the same world point stays under the mouse
			setOffset({
				x: mouseX - baseOffsetX - worldX * newScale,
				y: mouseY - baseOffsetY - worldY * newScale,
			});
			setScale(newScale);
		},
		[scale, offset, imageWidth, imageHeight],
	);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		canvas.addEventListener("wheel", handleWheel, { passive: false });
		return () => canvas.removeEventListener("wheel", handleWheel);
	}, [handleWheel]);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent<HTMLCanvasElement>) => {
			if (e.button === 0) {
				// Left click
				setIsPanning(true);
				setStartPan({ x: e.clientX - offset.x, y: e.clientY - offset.y });
			}
		},
		[offset],
	);

	const handleMouseMove = useCallback(
		(e: React.MouseEvent<HTMLCanvasElement>) => {
			if (isPanning) {
				setOffset({
					x: e.clientX - startPan.x,
					y: e.clientY - startPan.y,
				});
			}
		},
		[isPanning, startPan],
	);

	const handleMouseUp = useCallback(() => {
		setIsPanning(false);
	}, []);

	const handleMouseLeave = useCallback(() => {
		setIsPanning(false);
	}, []);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas || !imageSource) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		// Set canvas size to match container for display
		const container = containerRef.current;
		if (container) {
			const rect = container.getBoundingClientRect();
			canvas.width = rect.width;
			canvas.height = rect.height;
		}

		// Clear canvas
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		// Calculate scaling to fit image in canvas while maintaining aspect ratio
		const scaleX = canvas.width / imageWidth;
		const scaleY = canvas.height / imageHeight;
		const fitScale = Math.min(scaleX, scaleY);
		const displayWidth = imageWidth * fitScale;
		const displayHeight = imageHeight * fitScale;
		const baseOffsetX = (canvas.width - displayWidth) / 2;
		const baseOffsetY = (canvas.height - displayHeight) / 2;

		// Apply transform for zoom and pan
		ctx.save();
		ctx.translate(baseOffsetX + offset.x, baseOffsetY + offset.y);
		ctx.scale(scale, scale);

		// Draw image at origin
		ctx.drawImage(imageSource, 0, 0, displayWidth, displayHeight);

		const directionMode = showDirection
			? getDirectionDisplayMode(
					scale,
					getDetailScaleThreshold(detections, fitScale),
					directionModeRef.current,
				)
			: null;
		if (directionMode) directionModeRef.current = directionMode;

		// Draw detections
		for (const det of detections) {
			const color = COLORS[det.classId % COLORS.length];
			const corners = getOBBCorners(det);

			// Scale corners to display coordinates
			const scaledCorners = corners.map(([x, y]) => [
				x * fitScale,
				y * fitScale,
			]);

			// Draw OBB polygon
			if (showBoxes) {
				ctx.beginPath();
				ctx.moveTo(scaledCorners[0][0], scaledCorners[0][1]);
				for (let i = 1; i < scaledCorners.length; i++) {
					ctx.lineTo(scaledCorners[i][0], scaledCorners[i][1]);
				}
				ctx.closePath();
				ctx.strokeStyle = hexToRgba(color, 0.8);
				// Divide by scale so line width stays constant on screen when zooming
				ctx.lineWidth =
					Math.max(2, Math.min(displayWidth, displayHeight) / 500) / scale;
				ctx.stroke();
			}

			if (directionMode === "detail") {
				const longAxisLength = Math.max(det.width, det.height) * fitScale;
				drawDirectionMarker(ctx, {
					cx: det.cx * fitScale,
					cy: det.cy * fitScale,
					angle: getOBBLongAxisAngle(det),
					length: Math.max(0, longAxisLength * 0.4 - 4 / scale),
					lineWidth:
						Math.max(2, Math.min(displayWidth, displayHeight) / 500) / scale,
					color,
					alpha: 0.8,
				});
			}

			// Draw label inside OBB with rotation
			if (showLabels) {
				const label = `${det.className} ${(det.confidence * 100).toFixed(0)}%`;
				// Divide by scale so font size stays constant on screen when zooming
				const fontSize =
					Math.max(12, Math.min(displayWidth, displayHeight) / 80) / scale;
				ctx.font = `bold ${fontSize}px sans-serif`;
				const textMetrics = ctx.measureText(label);
				const textW = textMetrics.width;
				const asc = textMetrics.actualBoundingBoxAscent;
				const desc = textMetrics.actualBoundingBoxDescent;
				// pad is in canvas units: dividing by scale keeps it constant in screen pixels
				const pad = 3 / scale;
				const boxW = textW + 2 * pad;
				const boxH = asc + desc + 2 * pad;

				// Scale center coordinates
				const cx = det.cx * fitScale;
				const cy = det.cy * fitScale;

				// Save context state
				ctx.save();

				// Move to OBB center and rotate
				ctx.translate(cx, cy);
				ctx.rotate(det.angle);

				// Draw label background centered at OBB center (slightly transparent)
				ctx.fillStyle = hexToRgba(color, 0.75);
				ctx.fillRect(-boxW / 2, -boxH / 2, boxW, boxH);
				ctx.fillStyle = "rgba(255,255,255,0.95)";
				// Baseline offset to visually center text in box
				ctx.fillText(label, -textW / 2, (asc - desc) / 2);

				// Restore context state
				ctx.restore();
			}
		}

		if (directionMode === "overview") {
			const gridSize =
				DIRECTION_OVERVIEW_GRID_SPACING / Math.max(fitScale * scale, 1e-6);
			for (const group of aggregateDirectionsByGrid(detections, gridSize)) {
				drawDirectionMarker(ctx, {
					cx: group.x * fitScale,
					cy: group.y * fitScale,
					angle: group.angle,
					length: 24 / scale,
					lineWidth: 2 / scale,
					color: OVERVIEW_DIRECTION_COLOR,
					alpha: Math.min(0.9, 0.45 + Math.sqrt(group.count) * 0.1),
				});
			}
		}

		ctx.restore();
	}, [
		imageSource,
		detections,
		imageWidth,
		imageHeight,
		scale,
		offset,
		showBoxes,
		showLabels,
		showDirection,
	]);

	if (!imageSource) return null;

	return (
		// biome-ignore lint/a11y/noStaticElementInteractions: drag and drop is the intended interaction with keyboard support
		<div
			ref={containerRef}
			onDragOver={handleDragOver}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			onKeyDown={handleKeyDown}
			className={`relative ${onFileSelect && !disabled ? "cursor-pointer" : ""}`}
			{...(onFileSelect && !disabled ? { role: "button", tabIndex: 0 } : {})}
		>
			<canvas
				ref={canvasRef}
				className="max-h-[70vh] w-full object-contain"
				style={{
					imageRendering: "auto",
					cursor: isPanning ? "grabbing" : "grab",
				}}
				onMouseDown={handleMouseDown}
				onMouseMove={handleMouseMove}
				onMouseUp={handleMouseUp}
				onMouseLeave={handleMouseLeave}
			/>
			{onFileSelect && !disabled && (
				<div
					className={`pointer-events-none absolute inset-0 flex items-center justify-center transition-all ${
						isDragOver ? "bg-blue-500/20 backdrop-blur-sm" : "bg-transparent"
					}`}
				>
					{isDragOver && (
						<div className="rounded-lg bg-white/90 px-6 py-4 text-center shadow-lg dark:bg-gray-800/90">
							<p className="text-lg font-medium text-gray-800 dark:text-gray-200">
								次の画像をドロップ
							</p>
							<p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
								検出を実行します
							</p>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
