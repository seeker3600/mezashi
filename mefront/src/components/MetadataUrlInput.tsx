import { useState } from "react";

interface MetadataUrlInputProps {
	value: string;
	onChange: (url: string) => void;
	disabled: boolean;
}

export function MetadataUrlInput({
	value,
	onChange,
	disabled,
}: MetadataUrlInputProps) {
	const [draft, setDraft] = useState(value);

	const commit = () => {
		const trimmed = draft.trim();
		if (trimmed && trimmed !== value) {
			onChange(trimmed);
		}
	};

	return (
		<div className="mb-4 flex flex-col gap-1">
			<label
				htmlFor="metadata-url"
				className="text-xs font-medium text-gray-600 dark:text-gray-400"
			>
				モデルメタデータ URL（JSON ファイル）
			</label>
			<div className="flex gap-2">
				<input
					id="metadata-url"
					type="url"
					value={draft}
					onChange={(e) => setDraft(e.target.value)}
					onBlur={commit}
					onKeyDown={(e) => {
						if (e.key === "Enter") {
							e.currentTarget.blur();
						}
					}}
					disabled={disabled}
					placeholder="https://example.com/model.json"
					className="min-w-0 flex-1 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-800 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:placeholder-gray-500"
				/>
				<button
					type="button"
					onClick={commit}
					disabled={disabled || draft.trim() === value}
					className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-blue-500 dark:hover:bg-blue-600"
				>
					適用
				</button>
			</div>
		</div>
	);
}
