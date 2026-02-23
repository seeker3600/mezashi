/**
 * Minimal ONNX ModelProto metadata_props parser.
 *
 * Reads the binary protobuf to extract the `training_data_license`
 * key-value pair from the top-level field 14 (metadata_props).
 *
 * Protobuf field layout of ModelProto (relevant fields only):
 *   field  7 – graph       (LEN, large – skipped)
 *   field 14 – metadata_props (LEN, repeated StringStringEntryProto)
 *
 * StringStringEntryProto:
 *   field 1 – key   (LEN / string)
 *   field 2 – value (LEN / string)
 */

/** Model metadata extracted from ONNX metadata_props */
export interface ModelMetadata {
	trainingDataLicense?: string;
}

const MODEL_PATH = "/models/yolo26n-obb.onnx";
const METADATA_KEY = "training_data_license";

// Protobuf wire types
const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

/** Read a protobuf varint from *data* at *pos*. Returns [value, next_pos]. */
function readVarint(data: Uint8Array, pos: number): [number, number] {
	let result = 0;
	let shift = 0;
	while (pos < data.length) {
		const byte = data[pos++];
		// Use multiplication to avoid bitwise truncation to 32 bits
		result += (byte & 0x7f) * 2 ** shift;
		shift += 7;
		if ((byte & 0x80) === 0) break;
	}
	return [result, pos];
}

/** Decode UTF-8 bytes from a subarray slice. */
function decodeString(data: Uint8Array, pos: number, len: number): string {
	return new TextDecoder().decode(data.subarray(pos, pos + len));
}

/**
 * Parse a single StringStringEntryProto and return [key, value].
 * *data* must be the slice that represents exactly one entry.
 */
function parseStringStringEntry(data: Uint8Array): [string, string] {
	let pos = 0;
	let key = "";
	let value = "";
	while (pos < data.length) {
		let tag: number;
		[tag, pos] = readVarint(data, pos);
		const fieldNumber = Math.floor(tag / 8);
		const wireType = tag & 0x7;
		if (wireType === WIRE_LEN) {
			let len: number;
			[len, pos] = readVarint(data, pos);
			if (fieldNumber === 1) key = decodeString(data, pos, len);
			else if (fieldNumber === 2) value = decodeString(data, pos, len);
			pos += len;
		} else {
			// Unexpected wire type in StringStringEntryProto – stop
			break;
		}
	}
	return [key, value];
}

/**
 * Scan the top-level fields of an ONNX ModelProto binary and return
 * the extracted metadata.  The graph field is skipped in O(1) (just a
 * pointer advance), so performance is independent of model size.
 */
export function parseOnnxMetadata(buffer: ArrayBuffer): ModelMetadata {
	const data = new Uint8Array(buffer);
	const metadata: ModelMetadata = {};
	let pos = 0;

	while (pos < data.length) {
		let tag: number;
		[tag, pos] = readVarint(data, pos);
		const fieldNumber = Math.floor(tag / 8);
		const wireType = tag & 0x7;

		if (wireType === WIRE_VARINT) {
			// Skip varint value (read but discard)
			while (pos < data.length && data[pos++] & 0x80) {
				/* advance */
			}
		} else if (wireType === WIRE_LEN) {
			let len: number;
			[len, pos] = readVarint(data, pos);
			if (fieldNumber === 14) {
				// metadata_props – parse key/value
				const [key, value] = parseStringStringEntry(
					data.subarray(pos, pos + len),
				);
				if (key === METADATA_KEY) {
					metadata.trainingDataLicense = value;
				}
			}
			pos += len;
		} else if (wireType === WIRE_I64) {
			pos += 8;
		} else if (wireType === WIRE_I32) {
			pos += 4;
		} else {
			// Unknown wire type – stop
			break;
		}
	}

	return metadata;
}

let metadataPromise: Promise<ModelMetadata> | null = null;

/**
 * Load (or return cached) model metadata extracted from the ONNX file.
 * The browser will serve the model from cache once it has been loaded
 * by the inference session, so no extra network round-trip occurs.
 */
export function loadModelMetadata(): Promise<ModelMetadata> {
	if (!metadataPromise) {
		metadataPromise = fetch(MODEL_PATH)
			.then((res) => {
				if (!res.ok) return {};
				return res.arrayBuffer();
			})
			.then((bufferOrEmpty) => {
				if (!(bufferOrEmpty instanceof ArrayBuffer)) return {};
				return parseOnnxMetadata(bufferOrEmpty);
			})
			.catch(() => ({}));
	}
	return metadataPromise;
}
