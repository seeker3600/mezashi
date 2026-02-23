import type { AppStatus } from "../lib/appState";
import { LoadingSpinner } from "./LoadingSpinner";

export function StatusMessage({ status }: { status: AppStatus }) {
	if (status.type === "idle") return null;

	const isActive = status.type === "loading" || status.type === "processing";

	return (
		<div className="flex items-center gap-2">
			{isActive && <LoadingSpinner size="sm" />}
			<span
				className={
					isActive
						? "text-sm font-medium text-gray-700 dark:text-gray-300"
						: "text-sm text-gray-500 dark:text-gray-400"
				}
			>
				{status.message}
			</span>
		</div>
	);
}
