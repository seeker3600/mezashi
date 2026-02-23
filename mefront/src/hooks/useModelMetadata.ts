import { useEffect, useState } from "react";
import { loadModelMetadata, type ModelMetadata } from "../lib/modelMetadata";

/**
 * Load ONNX model metadata once at mount time.
 * Returns the metadata object (fields may be undefined while loading or on error).
 */
export function useModelMetadata(): ModelMetadata {
	const [metadata, setMetadata] = useState<ModelMetadata>({});

	useEffect(() => {
		let cancelled = false;
		loadModelMetadata().then((m) => {
			if (!cancelled) setMetadata(m);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	return metadata;
}
