import { useEffect, useRef } from "react";
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
}

export function DetectionCanvas({
	imageSource,
	detections,
	imageWidth,
	imageHeight,
}: DetectionCanvasProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);

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
		<canvas
			ref={canvasRef}
			className="max-h-[70vh] w-full object-contain"
			style={{ imageRendering: "auto" }}
		/>
	);
}
