interface LoadingSpinnerProps {
	size?: "sm" | "md" | "lg";
}

export function LoadingSpinner({ size = "md" }: LoadingSpinnerProps) {
	const sizeClasses = {
		sm: "h-4 w-4 border-2",
		md: "h-8 w-8 border-4",
		lg: "h-12 w-12 border-4",
	};

	return (
		<div
			className={`${sizeClasses[size]} animate-spin rounded-full border-solid border-blue-600 border-t-transparent`}
		/>
	);
}
