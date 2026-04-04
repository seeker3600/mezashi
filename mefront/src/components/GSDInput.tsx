import { useState } from "react";

/** Preset GSD values (m/px) shown in the dropdown */
const GSD_PRESETS = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0];

interface GSDDialogProps {
	/** Expected GSD from model metadata, used as the default value */
	modelDefault?: number;
	/** Previously used GSD value; takes precedence over modelDefault */
	initialValue?: number | null;
	/** Called with the confirmed GSD (null = skip, proceed without GSD) */
	onConfirm: (gsd: number | null) => void;
	/** Called when the user cancels the entire upload */
	onCancel: () => void;
}

export function GSDDialog({
	modelDefault,
	initialValue,
	onConfirm,
	onCancel,
}: GSDDialogProps) {
	const defaultStr =
		initialValue != null
			? String(initialValue)
			: modelDefault != null
				? String(modelDefault)
				: "";
	const [inputStr, setInputStr] = useState(defaultStr);

	const numVal = Number(inputStr);
	const isValid = inputStr !== "" && !Number.isNaN(numVal) && numVal >= 0.1;

	// Sync select to input when the value matches a preset exactly
	const selectVal =
		isValid && GSD_PRESETS.includes(numVal) ? String(numVal) : "";

	const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
		if (e.target.value !== "") {
			setInputStr(e.target.value);
		}
	};

	return (
		<div
			role="dialog"
			aria-modal="true"
			aria-labelledby="gsd-dialog-title"
			className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
		>
			<div className="w-full max-w-sm rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800">
				<h3
					id="gsd-dialog-title"
					className="mb-2 text-base font-semibold text-gray-900 dark:text-gray-100"
				>
					地上解像度を指定
				</h3>
				<p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
					PNG
					画像には地上解像度情報が含まれていません。解像度を指定すると縮小・タイル推論が最適化されます。
				</p>

				<div className="space-y-3">
					<div>
						<label
							htmlFor="gsd-preset"
							className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>
							プリセット
						</label>
						<select
							id="gsd-preset"
							value={selectVal}
							onChange={handlePresetChange}
							className="w-full rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-800 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
						>
							<option value="">カスタム値</option>
							{GSD_PRESETS.map((v) => (
								<option key={v} value={String(v)}>
									{v} m/px
								</option>
							))}
						</select>
					</div>

					<div>
						<label
							htmlFor="gsd-custom"
							className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>
							値 (m/px)
						</label>
						<input
							id="gsd-custom"
							type="number"
							min="0.1"
							step="0.1"
							value={inputStr}
							onChange={(e) => setInputStr(e.target.value)}
							placeholder={
								modelDefault != null ? `例: ${modelDefault}` : "例: 0.5"
							}
							className="w-full rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-800 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:placeholder:text-gray-500"
						/>
						{modelDefault != null && (
							<p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
								モデルデフォルト: {modelDefault} m/px
							</p>
						)}
					</div>
				</div>

				<div className="mt-6 flex justify-end gap-2">
					<button
						type="button"
						onClick={onCancel}
						className="rounded-md px-3 py-2 text-sm font-medium text-gray-600 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
					>
						キャンセル
					</button>
					<button
						type="button"
						onClick={() => onConfirm(null)}
						className="rounded-md px-3 py-2 text-sm font-medium text-gray-600 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
					>
						スキップ
					</button>
					<button
						type="button"
						onClick={() => onConfirm(numVal)}
						disabled={!isValid}
						className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
					>
						この解像度で推論
					</button>
				</div>
			</div>
		</div>
	);
}
