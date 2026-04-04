import type { DetectionTask, ModelMetadata } from "./types";

const VALID_TASKS: readonly DetectionTask[] = ["obb", "detect"];

/**
 * Fetch and validate a model metadata JSON file from the given URL.
 * Throws a descriptive error if the URL cannot be fetched or the JSON is invalid.
 */
export async function fetchModelMetadata(url: string): Promise<ModelMetadata> {
	let res: Response;
	try {
		res = await fetch(url);
	} catch (err) {
		throw new Error(
			`メタデータの取得に失敗しました (${url}): ${err instanceof Error ? err.message : String(err)}`,
		);
	}

	if (!res.ok) {
		throw new Error(
			`メタデータの取得に失敗しました (${res.status} ${res.statusText}): ${url}`,
		);
	}

	let json: unknown;
	try {
		json = await res.json();
	} catch {
		throw new Error("メタデータファイルが有効な JSON ではありません");
	}

	validateModelMetadata(json);
	return json;
}

export function validateModelMetadata(
	data: unknown,
): asserts data is ModelMetadata {
	if (typeof data !== "object" || data === null) {
		throw new Error("メタデータはオブジェクトである必要があります");
	}

	const obj = data as Record<string, unknown>;

	if (typeof obj.onnxUrl !== "string" || !obj.onnxUrl) {
		throw new Error(
			'メタデータの "onnxUrl" は空でない文字列である必要があります',
		);
	}

	if (typeof obj.inputSize !== "number" || obj.inputSize <= 0) {
		throw new Error(
			'メタデータの "inputSize" は 0 より大きい数値である必要があります',
		);
	}

	if (
		!Array.isArray(obj.labels) ||
		obj.labels.length === 0 ||
		!obj.labels.every((l) => typeof l === "string")
	) {
		throw new Error(
			'メタデータの "labels" は空でない文字列配列である必要があります',
		);
	}

	if (typeof obj.license !== "object" || obj.license === null) {
		throw new Error(
			'メタデータの "license" はオブジェクトである必要があります',
		);
	}

	const license = obj.license as Record<string, unknown>;
	if (typeof license.name !== "string" || !license.name) {
		throw new Error(
			'メタデータの "license.name" は空でない文字列である必要があります',
		);
	}

	if (typeof obj.name !== "string" || !obj.name) {
		throw new Error('メタデータの "name" は空でない文字列である必要があります');
	}

	if (
		typeof obj.task !== "string" ||
		!VALID_TASKS.includes(obj.task as DetectionTask)
	) {
		throw new Error(
			`メタデータの "task" は ${VALID_TASKS.map((t) => `"${t}"`).join(" | ")} のいずれかである必要があります`,
		);
	}

	// Optional: validate merge rules
	if (obj.merge !== undefined) {
		if (typeof obj.merge !== "object" || obj.merge === null) {
			throw new Error(
				'メタデータの "merge" はオブジェクトである必要があります',
			);
		}

		const merge = obj.merge as Record<string, unknown>;
		const labels = obj.labels as string[];

		for (const [key, value] of Object.entries(merge)) {
			if (
				!Array.isArray(value) ||
				value.length === 0 ||
				!value.every((v) => typeof v === "string")
			) {
				throw new Error(
					`メタデータの "merge.${key}" は空でない文字列配列である必要があります`,
				);
			}

			for (const label of value as string[]) {
				if (!labels.includes(label)) {
					throw new Error(
						`メタデータの "merge.${key}" に含まれる "${label}" は "labels" に存在しません`,
					);
				}
			}
		}
	}

	// Optional: validate expectedResolution
	if (obj.expectedResolution !== undefined) {
		if (
			typeof obj.expectedResolution !== "number" ||
			obj.expectedResolution <= 0
		) {
			throw new Error(
				'メタデータの "expectedResolution" は 0 より大きい数値（m/px）である必要があります',
			);
		}
	}
}
