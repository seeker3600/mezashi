import { useCallback, useState } from "react";

interface DropZoneProps {
	onFileSelect: (file: File) => void;
	disabled?: boolean;
}

export function DropZone({ onFileSelect, disabled }: DropZoneProps) {
	const [isDragOver, setIsDragOver] = useState(false);

	const handleDragOver = useCallback(
		(e: React.DragEvent) => {
			e.preventDefault();
			if (!disabled) setIsDragOver(true);
		},
		[disabled],
	);

	const handleDragLeave = useCallback(() => {
		setIsDragOver(false);
	}, []);

	const handleDrop = useCallback(
		(e: React.DragEvent) => {
			e.preventDefault();
			setIsDragOver(false);
			if (disabled) return;
			const file = e.dataTransfer.files[0];
			if (file) onFileSelect(file);
		},
		[disabled, onFileSelect],
	);

	const handleClick = useCallback(() => {
		if (disabled) return;
		const input = document.createElement("input");
		input.type = "file";
		input.accept = "image/*,.tif,.tiff";
		input.onchange = () => {
			const file = input.files?.[0];
			if (file) onFileSelect(file);
		};
		input.click();
	}, [disabled, onFileSelect]);

	return (
		<button
			type="button"
			onDragOver={handleDragOver}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			onClick={handleClick}
			disabled={disabled}
			className={`flex min-h-48 w-full cursor-pointer items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
				isDragOver
					? "border-blue-400 bg-blue-50 dark:bg-blue-950"
					: "border-gray-300 hover:border-gray-400 dark:border-gray-600 dark:hover:border-gray-500"
			} ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
		>
			<div className="text-center">
				<p className="text-lg font-medium text-gray-700 dark:text-gray-300">
					画像をドラッグ＆ドロップ
				</p>
				<p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
					またはクリックしてファイルを選択
				</p>
				<p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
					JPG, PNG, GeoTIFF に対応
				</p>
			</div>
		</button>
	);
}
