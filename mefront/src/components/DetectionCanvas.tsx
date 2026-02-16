import { useCallback, useEffect, useRef, useState } from "react";
import { getOBBCorners } from "../lib/obbUtils";
import type { Detection } from "../lib/types";

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

interface DetectionCanvasProps {
	imageSource: HTMLCanvasElement | HTMLImageElement | null;
	detections: Detection[];
	imageWidth: number;
	imageHeight: number;
	onFileSelect?: (file: File) => void;
	disabled?: boolean;
}

export function DetectionCanvas({
	imageSource,
	detections,
	imageWidth,
	imageHeight,
	onFileSelect,
	disabled = false,
}: DetectionCanvasProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [isDragOver, setIsDragOver] = useState(false);
	const [scale, setScale] = useState(1);
	const [offset, setOffset] = useState({ x: 0, y: 0 });
	const [isPanning, setIsPanning] = useState(false);
	const [startPan, setStartPan] = useState({ x: 0, y: 0 });

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
			const file = e.dataTransfer.files[0];
			if (file) onFileSelect(file);
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
				input.onchange = () => {
					const file = input.files?.[0];
					if (file) onFileSelect(file);
				};
				input.click();
			}
		},
		[disabled, onFileSelect],
	);

	// Reset zoom and pan when image changes
	useEffect(() => {
		setScale(1);
		setOffset({ x: 0, y: 0 });
	}, [imageSource, imageWidth, imageHeight]);

	const handleWheel = useCallback(
		(e: React.WheelEvent<HTMLCanvasElement>) => {
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
			ctx.beginPath();
			ctx.moveTo(scaledCorners[0][0], scaledCorners[0][1]);
			for (let i = 1; i < scaledCorners.length; i++) {
				ctx.lineTo(scaledCorners[i][0], scaledCorners[i][1]);
			}
			ctx.closePath();
			ctx.strokeStyle = color;
			ctx.lineWidth = Math.max(2, Math.min(displayWidth, displayHeight) / 500);
			ctx.stroke();

			// Draw label inside OBB with rotation
			const label = `${det.className} ${(det.confidence * 100).toFixed(0)}%`;
			const fontSize = Math.max(12, Math.min(displayWidth, displayHeight) / 80);
			ctx.font = `bold ${fontSize}px sans-serif`;
			const textMetrics = ctx.measureText(label);
			const textW = textMetrics.width + 6;
			const textH = fontSize + 4;

			// Scale center coordinates
			const cx = det.cx * fitScale;
			const cy = det.cy * fitScale;

			// Save context state
			ctx.save();

			// Move to OBB center and rotate
			ctx.translate(cx, cy);
			ctx.rotate(det.angle);

			// Draw label background centered at OBB center
			ctx.fillStyle = color;
			ctx.fillRect(-textW / 2, -textH / 2, textW, textH);
			ctx.fillStyle = "#fff";
			ctx.fillText(label, -textW / 2 + 3, textH / 2 - 4);

			// Restore context state
			ctx.restore();
		}

		ctx.restore();
	}, [imageSource, detections, imageWidth, imageHeight, scale, offset]);

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
				onWheel={handleWheel}
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
