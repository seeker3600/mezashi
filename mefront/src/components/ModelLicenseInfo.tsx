import { useState } from "react";
import type { ModelLicense } from "../lib/types";

interface ModelLicenseInfoProps {
	license: ModelLicense | undefined;
}

export function ModelLicenseInfo({ license }: ModelLicenseInfoProps) {
	const [textExpanded, setTextExpanded] = useState(false);

	if (!license) return null;

	return (
		<div className="mb-4 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400">
			<div className="flex flex-wrap items-center gap-x-2 gap-y-1">
				<span className="font-medium">モデルライセンス: </span>
				{license.url ? (
					<a
						href={license.url}
						target="_blank"
						rel="noopener noreferrer"
						className="underline hover:text-gray-800 dark:hover:text-gray-200"
					>
						{license.name}
					</a>
				) : (
					<span>{license.name}</span>
				)}

				<span className="text-red-500 dark:text-red-500">
					— ライセンスによっては商用利用が制限される場合があります
				</span>

				{license.text && (
					<button
						type="button"
						onClick={() => setTextExpanded((v) => !v)}
						className="ml-auto text-xs text-gray-400 underline hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
					>
						{textExpanded ? "折りたたむ ▲" : "詳細 ▼"}
					</button>
				)}
			</div>

			{license.text && textExpanded && (
				<p className="mt-2 whitespace-pre-wrap text-gray-500 dark:text-gray-500">
					{license.text}
				</p>
			)}
		</div>
	);
}
