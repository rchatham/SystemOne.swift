import Foundation
import MLX
import MLXNN
import SystemOne

/// LFM2-VL's image settings: the SigLIP2 NaFlex tower (`vision_config`), the
/// multimodal projector, and the image processor (`processor_config.json`).
public struct LFM2VLVisionConfiguration: Equatable, Sendable {
    /// The resize filter `resample` names, in PIL's numbering.
    public enum Resampling: Int, Equatable, Sendable {
        case bilinear = 2
        case bicubic = 3
    }

    // The tower.
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    /// Encoder layers run: all of them, or up to `vision_feature_layer`.
    public let layers: Int
    public let layerNormEpsilon: Float
    /// Side of the learned position grid (`sqrt(num_patches)`).
    public let positionSide: Int
    public let patchSize: Int

    // The projector.
    public let downsampleFactor: Int
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorLayerNorm: Bool
    /// The text trunk's width, which the projector writes.
    public let textHiddenSize: Int
    /// The placeholder token image features replace (`<image>`).
    public let imageTokenID: Int

    // The image processor.
    public let minImageTokens: Int
    public let maxImageTokens: Int
    public let tileSize: Int
    public let imageSplitting: Bool
    public let useThumbnail: Bool
    public let maxPixelsTolerance: Double
    public let resampling: Resampling
    /// Per-channel mean and std applied after `rescaleFactor`.
    public let imageMean: [Double]
    public let imageStd: [Double]
    public let rescaleFactor: Double
    /// Whether the markup brackets images with `<|image_start|>`, tile tags, and `<|image_end|>`.
    public let imageSpecialTokens: Bool

    /// Reads `config.json` and the `image_processor` of `processor_config.json`
    /// (or `preprocessor_config.json`), with transformers' defaults for missing keys.
    public init(modelFolder: URL) throws {
        func object(_ name: String) throws -> [String: Any]? {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw LFM2ConfigurationError.invalid("\(name) is not a JSON object")
            }
            return value
        }
        guard let config = try object("config.json") else { throw LFM2ConfigurationError.missing("config.json") }
        let processor = try object("processor_config.json")?["image_processor"] as? [String: Any]
            ?? object("preprocessor_config.json") ?? [:]
        try self.init(json: config, processor: processor)
    }

    /// `json` is an LFM2-VL `config.json`; `processor` its image processor's settings.
    public init(json: [String: Any], processor: [String: Any]) throws {
        guard let vision = json["vision_config"] as? [String: Any] else {
            throw LFM2ConfigurationError.missing("vision_config")
        }
        let text = json["text_config"] as? [String: Any] ?? [:]
        func int(_ object: [String: Any], _ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = object[key] as? Int { return value }
            if let fallback { return fallback }
            throw LFM2ConfigurationError.missing(key)
        }
        func double(_ object: [String: Any], _ key: String, _ fallback: Double) -> Double {
            (object[key] as? NSNumber)?.doubleValue ?? fallback
        }
        func bool(_ object: [String: Any], _ key: String, _ fallback: Bool) -> Bool {
            object[key] as? Bool ?? fallback
        }
        func doubles(_ object: [String: Any], _ key: String) throws -> [Double] {
            guard let values = object[key] else { return [0.5, 0.5, 0.5] }
            guard let list = values as? [NSNumber], list.count == 3 else {
                throw LFM2ConfigurationError.invalid("\(key) needs three channel values")
            }
            return list.map(\.doubleValue)
        }

        let activation = vision["hidden_act"] as? String ?? "gelu_pytorch_tanh"
        guard activation == "gelu_pytorch_tanh" else { throw LFM2ConfigurationError.unsupported("vision hidden_act \(activation)") }
        let projectorActivation = json["projector_hidden_act"] as? String ?? "gelu"
        guard projectorActivation == "gelu" else {
            throw LFM2ConfigurationError.unsupported("projector_hidden_act \(projectorActivation)")
        }
        hiddenSize = try int(vision, "hidden_size")
        intermediateSize = try int(vision, "intermediate_size")
        attentionHeads = try int(vision, "num_attention_heads")
        let encoderLayers = try int(vision, "num_hidden_layers")
        let featureLayer = try int(json, "vision_feature_layer", -1)
        layers = featureLayer == -1 ? encoderLayers : (featureLayer >= 0 ? featureLayer + 1 : encoderLayers + featureLayer + 1)
        layerNormEpsilon = Float(double(vision, "layer_norm_eps", 1e-6))
        let positions = try int(vision, "num_patches", 256)
        positionSide = Int(Double(positions).squareRoot())
        patchSize = try int(vision, "patch_size", 16)

        downsampleFactor = try int(json, "downsample_factor", 2)
        projectorHiddenSize = try int(json, "projector_hidden_size", 2560)
        projectorBias = bool(json, "projector_bias", true)
        projectorLayerNorm = bool(json, "projector_use_layernorm", true)
        textHiddenSize = try int(text, "hidden_size")
        imageTokenID = try int(json, "image_token_id", try int(json, "image_token_index", 396))

        minImageTokens = try int(processor, "min_image_tokens", 64)
        maxImageTokens = try int(processor, "max_image_tokens", 256)
        tileSize = try int(processor, "tile_size", 512)
        imageSplitting = bool(processor, "do_image_splitting", true)
        useThumbnail = bool(processor, "use_thumbnail", true)
        maxPixelsTolerance = double(processor, "max_pixels_tolerance", 2)
        let resample = try int(processor, "resample", Resampling.bilinear.rawValue)
        guard let resampling = Resampling(rawValue: resample) else {
            throw LFM2ConfigurationError.unsupported("resample \(resample)")
        }
        self.resampling = resampling
        imageMean = try doubles(processor, "image_mean")
        imageStd = try doubles(processor, "image_std")
        rescaleFactor = double(processor, "rescale_factor", 1.0 / 255)
        imageSpecialTokens = bool(json, "use_image_special_tokens", true)

        guard hiddenSize % attentionHeads == 0, (1...encoderLayers).contains(layers) else {
            throw LFM2ConfigurationError.invalid("vision tower of \(hiddenSize) features, \(attentionHeads) heads, \(layers) layers")
        }
        guard positionSide * positionSide == positions else {
            throw LFM2ConfigurationError.invalid("num_patches \(positions) is not a square grid")
        }
        let encoderPatch = try int(processor, "encoder_patch_size", patchSize)
        let processorFactor = try int(processor, "downsample_factor", downsampleFactor)
        guard encoderPatch == patchSize, processorFactor == downsampleFactor, downsampleFactor == 2 else {
            throw LFM2ConfigurationError.unsupported(
                "encoder_patch_size \(encoderPatch) and downsample_factor \(processorFactor) with patches of \(patchSize)"
            )
        }
        guard tileSize % (patchSize * downsampleFactor) == 0 else {
            throw LFM2ConfigurationError.invalid("tile_size \(tileSize)")
        }
        let tiles = (try int(processor, "min_tiles", 2), try int(processor, "max_tiles", 10))
        guard !imageSplitting || tiles == (2, 10) else {
            // The grid order depends on CPython's set order, pinned here for the shipped range.
            throw LFM2ConfigurationError.unsupported("min_tiles \(tiles.0) and max_tiles \(tiles.1)")
        }
    }
}

/// One image's processed views: transformers' `Lfm2VlImageProcessor` layout.
public struct LFM2VLImageLayout: Equatable, Sendable {
    /// Tile columns and rows; 1 × 1 for a single view.
    public let columns: Int
    public let rows: Int
    /// The smart-resized size: the single view, or a tiled image's thumbnail.
    public let height: Int
    public let width: Int
    public let tileSize: Int
    /// Whether a tiled image ends with its thumbnail.
    public let thumbnail: Bool

    public var tiled: Bool { columns > 1 || rows > 1 }

    /// Crop sizes `(height, width)` in encoding order: tiles row by row, then
    /// the thumbnail; or the one resized view.
    public var crops: [[Int]] {
        guard tiled else { return [[height, width]] }
        let tiles = Array(repeating: [tileSize, tileSize], count: columns * rows)
        return thumbnail ? tiles + [[height, width]] : tiles
    }
}

/// LFM2-VL's image path, Apache-2.0, following transformers'
/// `Lfm2VlImageProcessor`, `Lfm2VlProcessor`, `Siglip2VisionModel`, and
/// `Lfm2VlMultiModalProjector`:
///
/// - the processor's smart resize to 64–256 image tokens, and above twice that
///   a 512-pixel tile grid of 2 to 10 tiles plus a thumbnail, resized with
///   torchvision's antialiased uint8 kernel (`resample`);
/// - the SigLIP2 NaFlex tower on each view's 16-pixel patches, its learned
///   position grid resampled to the view (antialiased bilinear, float32);
/// - a 2 × 2 pixel unshuffle, an optional layer norm, and a two-layer GELU
///   projector into the text trunk's width;
/// - the markup that stands for an image in the prompt, one `<image>` per
///   feature row.
public final class LFM2VLVision {
    public static let imageToken = "<image>"
    static let imageStart = "<|image_start|>"
    static let imageEnd = "<|image_end|>"
    static let thumbnailToken = "<|img_thumbnail|>"

    /// Tile grids `(columns, rows)` with 2 to 10 tiles, in the order CPython's
    /// `sorted(set(...), key=area)` visits them in `_target_ratios`.
    static let tileGrids: [(Int, Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3),
        (7, 1), (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10),
    ]

    /// A view's interleaved RGB bytes.
    public struct Crop: Equatable, Sendable {
        public let height: Int
        public let width: Int
        public let pixels: [UInt8]
    }

    private struct Layer {
        let norm1Weight: MLXArray, norm1Bias: MLXArray
        let query: Linear, key: Linear, value: Linear, output: Linear
        let norm2Weight: MLXArray, norm2Bias: MLXArray
        let expand: Linear, contract: Linear
    }

    public let configuration: LFM2VLVisionConfiguration
    public let dataType: DType
    private let patchEmbedding: Linear
    /// `[side, side, hidden]` learned positions, float32 for resampling.
    private let positionGrid: MLXArray
    private let layers: [Layer]
    private let postNormWeight: MLXArray
    private let postNormBias: MLXArray
    private let projectorNorm: (weight: MLXArray, bias: MLXArray)?
    private let projectorExpand: Linear
    private let projectorContract: Linear

    public convenience init(modelFolder: URL, dataType: DType = .float32) throws {
        try self.init(
            store: SafetensorsCheckpoint(modelFolder: modelFolder),
            configuration: LFM2VLVisionConfiguration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the tower and projector under transformers' names
    /// (`model.vision_tower.vision_model.`, `model.multi_modal_projector.`,
    /// with or without `model.`) or mlx-vlm's (`vision_tower.`,
    /// `multi_modal_projector.`); MLX-quantized linears load as such.
    public init(store: any TensorStore, configuration: LFM2VLVisionConfiguration, dataType: DType = .float32) throws {
        self.configuration = configuration
        self.dataType = dataType
        let c = configuration
        let d = c.hiddenSize
        let towerPrefixes = ["model.vision_tower.vision_model.", "vision_tower.vision_model.", "model.vision_tower.", "vision_tower."]
        guard let tower = towerPrefixes.first(where: { store.descriptors["\($0)embeddings.patch_embedding.weight"] != nil }) else {
            throw WeightsError.tensorSetMismatch(missing: ["vision_tower.vision_model.embeddings.patch_embedding.weight"], unexpected: [])
        }
        let projectorPrefixes = ["model.multi_modal_projector.", "multi_modal_projector."]
        guard let projector = projectorPrefixes.first(where: { store.descriptors["\($0)linear_1.weight"] != nil }) else {
            throw WeightsError.tensorSetMismatch(missing: ["multi_modal_projector.linear_1.weight"], unexpected: [])
        }

        func check(_ name: String, _ shape: [Int]) throws {
            let module = String(name.dropLast(".weight".count))
            if name.hasSuffix(".weight"), shape.count == 2, store.isQuantized(module) {
                _ = try store.quantizationParameters(of: module, inputWidth: shape[1])
                return
            }
            let actual = try store.requiredTensor(name).descriptor.shape
            guard actual == shape else { throw WeightsError.shapeMismatch(name: name, expected: shape, actual: actual) }
        }
        func linear(_ name: String, _ output: Int, _ input: Int, bias: Bool = true) throws -> Linear {
            try check("\(name).weight", [output, input])
            if bias { try check("\(name).bias", [output]) }
            return try store.linear(name, inputWidth: input, bias: bias, dataType: dataType)
        }
        func vector(_ name: String, _ width: Int) throws -> MLXArray {
            try check(name, [width])
            return try store.requiredTensor(name).mlxArray(dataType)
        }

        let patchWidth = 3 * c.patchSize * c.patchSize
        patchEmbedding = try linear("\(tower)embeddings.patch_embedding", d, patchWidth)
        let positionName = "\(tower)embeddings.position_embedding.weight"
        try check(positionName, [c.positionSide * c.positionSide, d])
        positionGrid = try store.requiredTensor(positionName).mlxArray(.float32).reshaped(c.positionSide, c.positionSide, d)
        layers = try (0..<c.layers).map { index -> Layer in
            let p = "\(tower)encoder.layers.\(index)."
            return Layer(
                norm1Weight: try vector("\(p)layer_norm1.weight", d),
                norm1Bias: try vector("\(p)layer_norm1.bias", d),
                query: try linear("\(p)self_attn.q_proj", d, d),
                key: try linear("\(p)self_attn.k_proj", d, d),
                value: try linear("\(p)self_attn.v_proj", d, d),
                output: try linear("\(p)self_attn.out_proj", d, d),
                norm2Weight: try vector("\(p)layer_norm2.weight", d),
                norm2Bias: try vector("\(p)layer_norm2.bias", d),
                expand: try linear("\(p)mlp.fc1", c.intermediateSize, d),
                contract: try linear("\(p)mlp.fc2", d, c.intermediateSize)
            )
        }
        postNormWeight = try vector("\(tower)post_layernorm.weight", d)
        postNormBias = try vector("\(tower)post_layernorm.bias", d)

        let unshuffled = d * c.downsampleFactor * c.downsampleFactor
        if c.projectorLayerNorm {
            projectorNorm = (try vector("\(projector)layer_norm.weight", unshuffled), try vector("\(projector)layer_norm.bias", unshuffled))
        } else {
            projectorNorm = nil
        }
        projectorExpand = try linear("\(projector)linear_1", c.projectorHiddenSize, unshuffled, bias: c.projectorBias)
        projectorContract = try linear("\(projector)linear_2", c.textHiddenSize, c.projectorHiddenSize, bias: c.projectorBias)
    }

    // MARK: - The processor

    /// Python's `round(number / factor) * factor` (half to even).
    static func roundByFactor(_ number: Int, _ factor: Int) -> Int {
        Int((Double(number) / Double(factor)).rounded(.toNearestOrEven)) * factor
    }

    /// `smart_resize`: `(height, width)` divisible by 32 within the token budget.
    static func smartResize(height: Int, width: Int, configuration c: LFM2VLVisionConfiguration) -> (height: Int, width: Int) {
        let factor = c.patchSize * c.downsampleFactor
        let pixelsPerToken = c.patchSize * c.patchSize * c.downsampleFactor * c.downsampleFactor
        let minimum = Double(c.minImageTokens * pixelsPerToken)
        let maximum = Double(c.maxImageTokens * pixelsPerToken)
        let (h0, w0) = (Double(height), Double(width))
        var h = max(factor, roundByFactor(height, factor))
        var w = max(factor, roundByFactor(width, factor))
        if Double(h * w) > maximum {
            let beta = (h0 * w0 / maximum).squareRoot()
            h = max(factor, Int((h0 / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((w0 / beta / Double(factor)).rounded(.down)) * factor)
        } else if Double(h * w) < minimum {
            let beta = (minimum / (h0 * w0)).squareRoot()
            h = Int((h0 * beta / Double(factor)).rounded(.up)) * factor
            w = Int((w0 * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// The processor's layout for an image of `width × height` pixels.
    public static func layout(width: Int, height: Int, configuration c: LFM2VLVisionConfiguration) -> LFM2VLImageLayout {
        let factor = c.patchSize * c.downsampleFactor
        let pixelsPerToken = c.patchSize * c.patchSize * c.downsampleFactor * c.downsampleFactor
        let roundedHeight = max(c.patchSize, roundByFactor(height, factor))
        let roundedWidth = max(c.patchSize, roundByFactor(width, factor))
        let large = Double(roundedHeight * roundedWidth) > Double(c.maxImageTokens * pixelsPerToken) * c.maxPixelsTolerance
        let resized = smartResize(height: height, width: width, configuration: c)
        var grid = (1, 1)
        if large && c.imageSplitting {
            let aspect = Double(width) / Double(height)
            let area = Double(width * height)
            var best = Double.infinity
            for ratio in tileGrids {
                let difference = abs(aspect - Double(ratio.0) / Double(ratio.1))
                if difference < best {
                    best = difference
                    grid = ratio
                } else if difference == best, area > 0.5 * Double(c.tileSize * c.tileSize * ratio.0 * ratio.1) {
                    grid = ratio
                }
            }
        }
        return LFM2VLImageLayout(
            columns: grid.0, rows: grid.1, height: resized.height, width: resized.width,
            tileSize: c.tileSize, thumbnail: c.useThumbnail
        )
    }

    /// Image tokens one view of `height × width` pixels yields.
    static func tokens(height: Int, width: Int, configuration c: LFM2VLVisionConfiguration) -> Int {
        let rows = (height / c.patchSize + c.downsampleFactor - 1) / c.downsampleFactor
        let columns = (width / c.patchSize + c.downsampleFactor - 1) / c.downsampleFactor
        return rows * columns
    }

    /// `Lfm2VlProcessor`'s expansion of one `<image>`: the image's placeholder
    /// tokens, with tile tags when `use_image_special_tokens` is set.
    public static func markup(_ layout: LFM2VLImageLayout, configuration c: LFM2VLVisionConfiguration) -> String {
        let special = c.imageSpecialTokens
        let perTile = tokens(height: c.tileSize, width: c.tileSize, configuration: c)
        let resized = tokens(height: layout.height, width: layout.width, configuration: c)
        var text = special ? imageStart : ""
        if layout.tiled {
            for row in 1...layout.rows {
                for column in 1...layout.columns {
                    if special { text += "<|img_row_\(row)_col_\(column)|>" }
                    text += String(repeating: imageToken, count: perTile)
                }
            }
            if layout.thumbnail {
                if special { text += thumbnailToken }
                text += String(repeating: imageToken, count: resized)
            }
        } else {
            text += String(repeating: imageToken, count: resized)
        }
        if special { text += imageEnd }
        return text
    }

    /// The feature rows (and `<image>` tokens) an image yields.
    public static func featureCount(_ layout: LFM2VLImageLayout, configuration c: LFM2VLVisionConfiguration) -> Int {
        layout.crops.reduce(0) { $0 + tokens(height: $1[0], width: $1[1], configuration: c) }
    }

    /// Every view of `image` as RGB bytes, resized as the processor resizes:
    /// the tiles of the grid-sized image row by row, then the thumbnail.
    public static func crops(of image: RGBImage, configuration c: LFM2VLVisionConfiguration) -> [Crop] {
        let plan = layout(width: image.width, height: image.height, configuration: c)
        let filter: AntialiasedResize.Filter = c.resampling == .bicubic ? .cubic : .triangle
        func resized(width: Int, height: Int) -> [UInt8] {
            AntialiasedResize.resize(
                image.pixels, width: image.width, height: image.height, channels: 3,
                toWidth: width, toHeight: height, filter: filter, kernel: .aten
            )
        }
        guard plan.tiled else {
            return [Crop(height: plan.height, width: plan.width, pixels: resized(width: plan.width, height: plan.height))]
        }
        let tile = c.tileSize
        let bigWidth = plan.columns * tile
        let big = resized(width: bigWidth, height: plan.rows * tile)
        var crops: [Crop] = []
        for row in 0..<plan.rows {
            for column in 0..<plan.columns {
                var pixels = [UInt8]()
                pixels.reserveCapacity(tile * tile * 3)
                for y in (row * tile)..<((row + 1) * tile) {
                    let start = (y * bigWidth + column * tile) * 3
                    pixels += big[start..<(start + tile * 3)]
                }
                crops.append(Crop(height: tile, width: tile, pixels: pixels))
            }
        }
        if plan.thumbnail {
            crops.append(Crop(height: plan.height, width: plan.width, pixels: resized(width: plan.width, height: plan.height)))
        }
        return crops
    }

    // MARK: - The tower and projector

    /// `[featureCount, textHidden]` features for one image: every view's
    /// projected patches, in the order its markup's `<image>` tokens run.
    public func callAsFunction(_ image: RGBImage) -> MLXArray {
        let crops = Self.crops(of: image, configuration: configuration)
        var outputs: [MLXArray] = []
        var start = 0
        // Consecutive views of one size (the tiles) run as one batch.
        while start < crops.count {
            var end = start + 1
            while end < crops.count, crops[end].height == crops[start].height, crops[end].width == crops[start].width {
                end += 1
            }
            outputs.append(encode(Array(crops[start..<end])))
            start = end
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    /// Projected features of same-sized views, `[views · tokens, textHidden]`.
    func encode(_ crops: [Crop]) -> MLXArray {
        let (height, width) = (crops[0].height, crops[0].width)
        let rows = height / configuration.patchSize
        let columns = width / configuration.patchSize
        let hidden = towerOutput(crops).reshaped(crops.count, rows, columns, configuration.hiddenSize)
        let projected = project(hidden)
        return projected.reshaped(-1, projected.dim(-1))
    }

    /// The tower's last hidden state for same-sized views: `[views, patches, hidden]`.
    func towerOutput(_ crops: [Crop]) -> MLXArray {
        let c = configuration
        let (height, width) = (crops[0].height, crops[0].width)
        let (rows, columns) = (height / c.patchSize, width / c.patchSize)
        let views = crops.count
        // Rescaled and normalized as the processor's fused step: (x - mean / r) / (std / r).
        let means = MLXArray(c.imageMean.map { Float($0 / c.rescaleFactor) })
        let stds = MLXArray(c.imageStd.map { Float($0 / c.rescaleFactor) })
        let pixels = MLXArray(crops.flatMap(\.pixels), [views, height, width, 3]).asType(.float32)
        let normalized = ((pixels - means) / stds).asType(dataType)
        // 16 × 16 patches flattened as (dy, dx, channel).
        let patches = normalized
            .reshaped(views, rows, c.patchSize, columns, c.patchSize, 3)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(views, rows * columns, 3 * c.patchSize * c.patchSize)
        return tower(patchEmbedding(patches) + positions(rows: rows, columns: columns))
    }

    /// The position grid resized to `rows × columns` (antialiased bilinear in
    /// float32, as ATen interpolates it on the CPU), flattened row-major.
    func positions(rows: Int, columns: Int) -> MLXArray {
        let side = positionGrid.dim(0)
        func matrix(_ input: Int, _ output: Int) -> MLXArray {
            let taps = AntialiasedResize.floatTaps(input: input, output: output)
            var values = [Float](repeating: 0, count: output * input)
            for (index, weights) in taps.weights.enumerated() {
                for (tap, weight) in weights.enumerated() { values[index * input + taps.starts[index] + tap] = weight }
            }
            return MLXArray(values, [output, input])
        }
        // [H, W, C]: width first, then height, as ATen's separable pass runs.
        let alongWidth = matmul(matrix(side, columns), positionGrid)
        let alongHeight = matmul(matrix(side, rows), alongWidth.swappedAxes(0, 1)).swappedAxes(0, 1)
        return alongHeight.reshaped(1, rows * columns, configuration.hiddenSize).asType(dataType)
    }

    /// The unpadded SigLIP2 encoder: pre-norm layers, then the post layer norm.
    private func tower(_ input: MLXArray) -> MLXArray {
        let c = configuration
        let eps = c.layerNormEpsilon
        let heads = c.attentionHeads
        let headDimension = c.hiddenSize / heads
        let (batch, length) = (input.dim(0), input.dim(1))
        func split(_ x: MLXArray) -> MLXArray {
            x.reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
        }
        var hidden = input
        for layer in layers {
            let normalized = MLXFast.layerNorm(hidden, weight: layer.norm1Weight, bias: layer.norm1Bias, eps: eps)
            let attended = MLXFast.scaledDotProductAttention(
                queries: split(layer.query(normalized)),
                keys: split(layer.key(normalized)),
                values: split(layer.value(normalized)),
                scale: 1 / Float(headDimension).squareRoot(),
                mask: nil
            )
            hidden = hidden + layer.output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
            let feedForwardInput = MLXFast.layerNorm(hidden, weight: layer.norm2Weight, bias: layer.norm2Bias, eps: eps)
            hidden = hidden + layer.contract(geluApproximate(layer.expand(feedForwardInput)))
        }
        return MLXFast.layerNorm(hidden, weight: postNormWeight, bias: postNormBias, eps: eps)
    }

    /// The 2 × 2 pixel unshuffle and the projector, `[views, rows/2 · columns/2, textHidden]`.
    private func project(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let factor = configuration.downsampleFactor
        let unshuffled = x
            .reshaped(batch, height, width / factor, channels * factor).transposed(0, 2, 1, 3)
            .reshaped(batch, width / factor, height / factor, channels * factor * factor).transposed(0, 2, 1, 3)
        var features = unshuffled
        if let norm = projectorNorm {
            features = MLXFast.layerNorm(features, weight: norm.weight, bias: norm.bias, eps: 1e-5)
        }
        let projected = projectorContract(gelu(projectorExpand(features)))
        return projected.reshaped(batch, -1, projected.dim(-1))
    }
}
