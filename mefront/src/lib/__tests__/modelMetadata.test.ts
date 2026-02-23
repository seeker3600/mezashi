import { describe, expect, it } from "vitest";
import { parseOnnxMetadata } from "../modelMetadata";

/**
 * Build a minimal protobuf binary that contains a ModelProto with only
 * the metadata_props field (field 14) to keep the test data small.
 *
 * Protobuf encoding helpers:
 *   varint: variable-length integer
 *   tag:    (field_number << 3) | wire_type
 *   LEN:    wire_type = 2 → followed by varint length then bytes
 */
function encodeVarint(value: number): number[] {
	const bytes: number[] = [];
	while (value > 0x7f) {
		bytes.push((value & 0x7f) | 0x80);
		value = Math.floor(value / 128);
	}
	bytes.push(value);
	return bytes;
}

function encodeUtf8(s: string): number[] {
	return [...new TextEncoder().encode(s)];
}

function encodeLenField(fieldNumber: number, bytes: number[]): number[] {
	const tag = (fieldNumber << 3) | 2; // wire type 2 = LEN
	return [...encodeVarint(tag), ...encodeVarint(bytes.length), ...bytes];
}

/** Build a StringStringEntryProto { key, value } */
function buildStringStringEntry(key: string, value: string): number[] {
	return [
		...encodeLenField(1, encodeUtf8(key)),
		...encodeLenField(2, encodeUtf8(value)),
	];
}

/** Wrap bytes as ModelProto field 14 (metadata_props) */
function buildModelProtoMetadata(
	entries: Array<[string, string]>,
): ArrayBuffer {
	const body: number[] = [];
	for (const [key, value] of entries) {
		const entry = buildStringStringEntry(key, value);
		body.push(...encodeLenField(14, entry));
	}
	return new Uint8Array(body).buffer;
}

describe("parseOnnxMetadata", () => {
	it("should return empty object when buffer is empty", () => {
		const result = parseOnnxMetadata(new ArrayBuffer(0));
		expect(result).toEqual({});
	});

	it("should extract training_data_license from metadata_props", () => {
		const buf = buildModelProtoMetadata([
			["training_data_license", "MIT License"],
		]);
		const result = parseOnnxMetadata(buf);
		expect(result.trainingDataLicense).toBe("MIT License");
	});

	it("should ignore unrecognised metadata keys", () => {
		const buf = buildModelProtoMetadata([
			["other_key", "other_value"],
			["training_data_license", "CC-BY 4.0"],
		]);
		const result = parseOnnxMetadata(buf);
		expect(result.trainingDataLicense).toBe("CC-BY 4.0");
	});

	it("should return undefined trainingDataLicense when key is absent", () => {
		const buf = buildModelProtoMetadata([["author", "me"]]);
		const result = parseOnnxMetadata(buf);
		expect(result.trainingDataLicense).toBeUndefined();
	});

	it("should handle a varint (ir_version) field before metadata_props", () => {
		// ir_version is field 1, wire type 0 (varint)
		const irVersionTag = encodeVarint((1 << 3) | 0); // tag
		const irVersionValue = encodeVarint(8); // value = 8
		const metadataEntry = buildStringStringEntry(
			"training_data_license",
			"Apache-2.0",
		);
		const metadataField = encodeLenField(14, metadataEntry);

		const body = new Uint8Array([
			...irVersionTag,
			...irVersionValue,
			...metadataField,
		]);
		const result = parseOnnxMetadata(body.buffer);
		expect(result.trainingDataLicense).toBe("Apache-2.0");
	});
});
