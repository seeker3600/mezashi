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

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas || !imageSource) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		// Set canvas size to match image
		canvas.width = imageWidth;
		canvas.height = imageHeight;

		// Draw image
		ctx.drawImage(imageSource, 0, 0, imageWidth, imageHeight);

		// Draw detections
		for (const det of detections) {
			const color = COLORS[det.classId % COLORS.length];
			const corners = getOBBCorners(det);

			// Draw OBB polygon
			ctx.beginPath();
			ctx.moveTo(corners[0][0], corners[0][1]);
			for (let i = 1; i < corners.length; i++) {
				ctx.lineTo(corners[i][0], corners[i][1]);
			}
			ctx.closePath();
			ctx.strokeStyle = color;
			ctx.lineWidth = Math.max(2, Math.min(imageWidth, imageHeight) / 500);
			ctx.stroke();

			// Draw label inside OBB with rotation
			const label = `${det.className} ${(det.confidence * 100).toFixed(0)}%`;
			const fontSize = Math.max(12, Math.min(imageWidth, imageHeight) / 80);
			ctx.font = `bold ${fontSize}px sans-serif`;
			const textMetrics = ctx.measureText(label);
			const textW = textMetrics.width + 6;
			const textH = fontSize + 4;

			// Save context state
			ctx.save();

			// Move to OBB center and rotate
			ctx.translate(det.cx, det.cy);
			ctx.rotate(det.angle);

			// Draw label background centered at OBB center
			ctx.fillStyle = color;
			ctx.fillRect(-textW / 2, -textH / 2, textW, textH);
			ctx.fillStyle = "#fff";
			ctx.fillText(label, -textW / 2 + 3, textH / 2 - 4);

			// Restore context state
			ctx.restore();
		}
	}, [imageSource, detections, imageWidth, imageHeight]);

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
				style={{ imageRendering: "auto" }}
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
								2枚目の画像をドロップ
							</p>
							<p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
								検出を実行します
							</p>
						</div>
					)}
				</div>
			)}
			{onFileSelect && !disabled && !isDragOver && (
				<div className="pointer-events-none absolute bottom-4 right-4 rounded-md bg-gray-800/70 px-3 py-2 text-xs text-white backdrop-blur-sm">
					2枚目の画像をドロップできます
				</div>
			)}
		</div>
	);
}
