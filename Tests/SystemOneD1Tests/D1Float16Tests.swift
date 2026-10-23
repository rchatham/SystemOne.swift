import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneD1

/// `precision: .float16` against the float32 Python fixture. Half precision is
/// not bit-faithful, so these check what the d1-omni model card promises: the
/// same top answer on every row, with probabilities close to float32's.
final class D1Float16Tests: XCTestCase {
    /// Largest probability difference allowed from the float32 reference.
    static let probabilityTolerance = 2e-2

    // MARK: Offline

    func testBlockedMaskValueFitsTheComputeDType() {
        XCTAssertEqual(D1Trunk.blocked(.float32), -1e9)
        XCTAssertEqual(D1Trunk.blocked(.float16), -65_504)
    }

    /// `mlxArray(as: .float16)` hands MLX a page-aligned buffer. Metal must wrap
    /// it in place (the finalizer runs only when the array is freed), not copy
    /// it, even when the length is not a whole number of pages.
    func testPageAlignedBufferIsWrappedWithoutACopy() throws {
        guard ProcessInfo.processInfo.environment["D1_DEVICE"] == "gpu" else { throw XCTSkip("Set D1_DEVICE=gpu to check Metal buffer wrapping.") }
        let page = Int(getpagesize())
        for byteCount in [page, 1_000, 3 * page + 6] {
            var buffer: UnsafeMutableRawPointer?
            XCTAssertEqual(posix_memalign(&buffer, page, (byteCount + page - 1) / page * page), 0)
            memset(buffer, 0, byteCount)
            var released = false
            var array: MLXArray? = MLXArray(rawPointer: buffer!, [byteCount / 2], dtype: .float16) {
                released = true
                free(buffer)
            }
            XCTAssertFalse(released, "\(byteCount) bytes were copied instead of wrapped")
            XCTAssertEqual(array!.sum().item(Float.self), 0)
            array = nil
            // Metal may hold the buffer until in-flight command buffers complete.
            Stream.gpu.synchronize()
            let deadline = Date().addingTimeInterval(2)
            while !released && Date() < deadline { RunLoop.current.run(until: Date().addingTimeInterval(0.01)) }
            print("\(byteCount) bytes released: \(released)")
            XCTAssertTrue(released, "\(byteCount) bytes were not released with the array")
        }
    }

    /// A float16 load refuses weights float16 would turn into infinity, both raw
    /// tensors and BatchNorm folded in float32, and still accepts values that are
    /// only ever used in float32. Runs a tiny synthetic checkpoint.
    func testFloat16LoadRejectsWeightsBeyondItsRange() throws {
        guard ProcessInfo.processInfo.environment["D1_DEVICE"] == "gpu" else {
            throw XCTSkip("Set D1_DEVICE=gpu to build MLX arrays from the Xcode test bundle.")
        }
        let layer = "audio.encoder.layers.0.conv.batch_norm"
        func load(_ overrides: [String: Float], _ precision: D1Precision) throws -> D1Model {
            let (folder, configuration) = try Self.tinyCheckpoint(overrides: overrides)
            defer { try? FileManager.default.removeItem(at: folder) }
            return try D1Model(weights: D1Weights(contentsOf: folder.appendingPathComponent("model.safetensors"),
                                                  configuration: configuration, precision: precision))
        }
        let raw = "encoder.layers.0.feed_forward.w1.weight"
        XCTAssertThrowsError(try load([raw: 70_000], .float16)) { error in
            XCTAssertEqual(error as? WeightsError, .float16Overflow(name: raw, maximumMagnitude: 70_000))
        }
        XCTAssertNoThrow(try load([raw: 70_000], .float32))
        // 300 / sqrt(0 + 1e-5) ≈ 94,868: every stored value fits, the folded scale does not.
        XCTAssertThrowsError(try load(["\(layer).weight": 300], .float16)) { error in
            guard case .float16Overflow(let name, _) = error as? WeightsError else { return XCTFail("\(error)") }
            XCTAssertEqual(name, "\(layer) (folded scale)")
        }
        // A huge running variance is only read in float32 and folds to a small scale.
        XCTAssertNoThrow(try load(["\(layer).weight": 1, "\(layer).running_var": 1e6], .float16))
    }

    /// A d1 checkpoint with every dimension shrunk, all zeros except `overrides`
    /// (each tensor's first element).
    static func tinyCheckpoint(overrides: [String: Float]) throws -> (URL, D1Configuration) {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let config = #"""
        {"max_length": 512, "image_text_length": 256, "audio_text_length": 256, "head_layers": 1,
         "projector_hidden_size": 32, "temperatures": {}, "bos_token_id": 1, "pad_token_id": 0,
         "text_config": {"vocab_size": 8, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
           "num_attention_heads": 1, "num_key_value_heads": 1, "layer_types": ["conv", "full_attention"],
           "norm_eps": 1e-5, "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 64,
           "rope_theta": 1000000.0},
         "vision_config": {"hidden_size": 32, "intermediate_size": 32, "layer_norm_eps": 1e-6,
           "num_attention_heads": 1, "num_channels": 3, "num_hidden_layers": 1, "num_patches": 4, "patch_size": 16},
         "audio_config": {"feat_in": 8, "n_layers": 1, "d_model": 16, "subsampling_conv_channels": 4,
           "ff_expansion_factor": 1, "n_heads": 1, "conv_kernel_size": 3, "residual_width": 8}}
        """#
        try Data(config.utf8).write(to: folder.appendingPathComponent("config.json"))
        let configuration = try D1Configuration(modelFolder: folder)

        var entries: [String] = []
        var payload = Data()
        for spec in D1Weights.schema(for: configuration).sorted(by: { $0.name < $1.name }) {
            let count = spec.shape.reduce(1, *)
            let start = payload.count
            if spec.dataType == .int64 {
                payload.append(Data(count: 8 * count))
            } else {
                var values = [Float](repeating: 0, count: count)
                if let value = overrides[spec.name] { values[0] = value }
                values.withUnsafeBytes { payload.append(contentsOf: $0) }
            }
            entries.append(#""\#(spec.name)":{"dtype":"\#(spec.dataType.rawValue)","shape":\#(spec.shape),"data_offsets":[\#(start),\#(payload.count)]}"#)
        }
        var header = Array("{\(entries.joined(separator: ","))}".utf8)
        while header.count % 8 != 0 { header.append(0x20) }
        var file = Data((0..<8).map { UInt8(truncatingIfNeeded: UInt64(header.count) >> UInt64($0 * 8)) })
        file.append(contentsOf: header)
        file.append(payload)
        try file.write(to: folder.appendingPathComponent("model.safetensors"))
        return (folder, configuration)
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    /// Every stage stays float16: a float32 constant would silently promote and
    /// bring back the float32 activation memory.
    func testActivationsStayFloat16() throws {
        let weights = try D1TextTests.loadWeights(precision: .float16)
        try Device.withDefaultDevice(D1TextTests.device) {
            let model = try D1Model(weights: weights)
            let fixture = try D1Fixture.load()
            let image = fixture.vision.requests[0].images[0]
            let audio = fixture.audio.requests[0].audio
            let mediaPrefixes: [SystemOneMedia] = [
                .images([syntheticImage(width: image.width, height: image.height, seed: image.seed)]),
                .audio(AudioClip(pcm16: syntheticAudio(samples: audio.samples, seed: audio.seed))),
            ]
            for media in mediaPrefixes {
                XCTAssertEqual(try model.encode(media).dtype, .float16, "\(media)")
            }

            let row = fixture.textCases[0].rows[0]
            let prefix = D1MediaPrefix(media: mediaPrefixes[1], length: D1Model.prefixLength(of: mediaPrefixes[1]))
            let rows = [
                D1Row(ids: row.ids, markers: row.markers, questionType: .init(rawValue: row.qtype)!, prefix: nil),
                D1Row(ids: row.ids, markers: row.markers, questionType: .init(rawValue: row.qtype)!, prefix: prefix),
            ]
            let embeddings = model.trunk.embedding(MLXArray(row.ids.map(Int32.init))).expandedDimensions(axis: 0)
            let padded = padded(embeddings, widths: [0, [0, 3], 0])
            XCTAssertEqual(model.trunk(padded, lengths: [row.ids.count], prefixLengths: [0]).output.dtype, .float16)
            XCTAssertEqual(try model.logits(for: rows).dtype, .float16)
        }
    }

    func testTextRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().textCases {
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 text worst probability error: \(worst)")
    }

    func testImageRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().vision.requests {
            let images = testCase.images.map { syntheticImage(width: $0.width, height: $0.height, seed: $0.seed) }
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .images(images)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 image worst probability error: \(worst)")
    }

    func testAudioRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().audio.requests {
            let clip = AudioClip(pcm16: syntheticAudio(samples: testCase.audio.samples, seed: testCase.audio.seed))
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .audio(clip)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 audio worst probability error: \(worst)")
    }

    private static func engine() async throws -> SystemOneEngine {
        let weights = try D1TextTests.loadWeights(precision: .float16)
        return SystemOneEngine(backend: D1Backend(
            modelID: "d1-omni-600M",
            configuration: weights.configuration,
            tokenizer: try await D1Tokenizer(modelFolder: try D1Fixture.modelFolder()),
            model: try D1Model(weights: weights),
            logitsProvider: nil
        ))
    }
}

/// Checks the top answer against the fixture and returns the largest
/// probability difference, failing past `D1Float16Tests.probabilityTolerance`.
@discardableResult
func assertTopAnswer(
    _ answer: Answer,
    matches expected: JSONValue,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) -> Double {
    func pairs(_ key: String) -> [Double] { expected[key]!.array.map { $0.pair.1.number! } }
    func argmax(_ p: [Double]) -> Int { p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) }! }
    let (actual, reference): ([Double], [Double])
    switch answer {
    case .noul(let noul):
        (actual, reference) = ([1 - noul.noul, noul.noul], [1 - expected["noul"]!.number!, expected["noul"]!.number!])
    case .choice(let choice):
        XCTAssertEqual(choice.choice, expected["choice"]?.string, message, file: file, line: line)
        (actual, reference) = (choice.probabilities.map(\.probability), pairs("probabilities"))
    case .score(let score):
        (actual, reference) = (score.probabilities.map(\.probability), pairs("probabilities"))
    }
    XCTAssertEqual(argmax(actual), argmax(reference), "\(message): top answer changed", file: file, line: line)
    let error = zip(actual, reference).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(error, D1Float16Tests.probabilityTolerance, message, file: file, line: line)
    return error
}
