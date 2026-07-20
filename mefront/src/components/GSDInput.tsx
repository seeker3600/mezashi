import { useState } from "react";

const SUGGESTED_GSDS = [0.1, 0.3, 1, 10, 30];

interface GSDDialogProps {
	/** Expected GSD from model metadata, used as the default value */
	modelDefault?: number;
	/** Previously used GSD value; takes precedence over modelDefault */
	initialValue?: number | null;
	/** Called with the confirmed GSD */
	onConfirm: (gsd: number) => void;
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
	const suggestedGSDs = modelDefault
		? [...new Set([...SUGGESTED_GSDS, modelDefault])].sort((a, b) => a - b)
		: SUGGESTED_GSDS;

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
					画像の大きさを指定
				</h3>
				<p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
					この PNG には地図上の距離情報がありません。画像の 1 ピクセルが現地で何
					m に相当するかを指定すると、モデルに適した大きさで検出します。
				</p>

				<div className="space-y-3">
					<div>
						<label
							htmlFor="gsd-custom"
							className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>
							1 ピクセルあたりの距離
						</label>
						<div className="flex rounded-md border border-gray-300 bg-white focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 dark:border-gray-600 dark:bg-gray-700">
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
								className="min-w-0 flex-1 rounded-l-md bg-transparent px-3 py-1.5 text-sm text-gray-800 outline-none placeholder:text-gray-400 dark:text-gray-200 dark:placeholder:text-gray-500"
							/>
							<span className="flex items-center border-l border-gray-300 px-3 text-sm text-gray-600 dark:border-gray-600 dark:text-gray-300">
								m
							</span>
						</div>
						{modelDefault != null && (
							<p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
								推奨: {modelDefault} m/px（このモデルの学習時の解像度）
							</p>
						)}
					</div>
					<div>
						<p className="mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">
							よく使う解像度
						</p>
						<div className="flex flex-wrap gap-2">
							{suggestedGSDs.map((gsd) => {
								const isRecommended = gsd === modelDefault;
								const label = gsd < 1 ? `${gsd * 100} cm` : `${gsd} m`;
								return (
									<button
										key={gsd}
										type="button"
										onClick={() => setInputStr(String(gsd))}
										className={`rounded-md border px-2.5 py-1 text-sm transition-colors ${
											numVal === gsd
												? "border-blue-600 bg-blue-600 text-white"
												: "border-gray-300 text-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-700"
										}`}
									>
										{label}
										{isRecommended && " (推奨)"}
									</button>
								);
							})}
						</div>
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
