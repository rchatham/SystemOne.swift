// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation

/// `config.json` of a d1-omni checkpoint. Shapes, layer order, context limits,
/// and calibration temperatures are all read from here rather than hard-coded.
public struct D1Configuration: Decodable, Equatable, Sendable {
    /// LFM2.5 encoder trunk.
    public struct Text: Decodable, Equatable, Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let layerTypes: [String]
        public let normEps: Float
        public let convLCache: Int
        public let blockFFNDimMultiplier: Double
        public let blockMultipleOf: Int
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case convLCache = "conv_L_cache"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockMultipleOf = "block_multiple_of"
            case ropeTheta = "rope_theta"
        }

        public var headDimension: Int { hiddenSize / numAttentionHeads }

        /// LFM2's SwiGLU width: two thirds of `intermediate_size`, scaled, then
        /// rounded up to a multiple of `block_multiple_of`.
        public var feedForwardSize: Int {
            let scaled = Int(blockFFNDimMultiplier * Double(2 * intermediateSize / 3))
            return blockMultipleOf * ((scaled + blockMultipleOf - 1) / blockMultipleOf)
        }

        public func isAttentionLayer(_ index: Int) -> Bool {
            layerTypes[index] == "full_attention"
        }
    }

    /// SigLIP2 NaFlex vision tower.
    public struct Vision: Decodable, Equatable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let layerNormEps: Float
        public let numAttentionHeads: Int
        public let numChannels: Int
        public let numHiddenLayers: Int
        public let numPatches: Int
        public let patchSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layerNormEps = "layer_norm_eps"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case numHiddenLayers = "num_hidden_layers"
            case numPatches = "num_patches"
            case patchSize = "patch_size"
        }
    }

    /// FastConformer audio encoder.
    public struct Audio: Decodable, Equatable, Sendable {
        public let featIn: Int
        public let nLayers: Int
        public let dModel: Int
        public let subsamplingConvChannels: Int
        public let ffExpansionFactor: Int
        public let nHeads: Int
        public let convKernelSize: Int
        public let residualWidth: Int

        enum CodingKeys: String, CodingKey {
            case featIn = "feat_in"
            case nLayers = "n_layers"
            case dModel = "d_model"
            case subsamplingConvChannels = "subsampling_conv_channels"
            case ffExpansionFactor = "ff_expansion_factor"
            case nHeads = "n_heads"
            case convKernelSize = "conv_kernel_size"
            case residualWidth = "residual_width"
        }

        /// Frequency bins left after three stride-2 subsampling convolutions.
        public var subsampledFeatures: Int {
            var features = featIn
            for _ in 0..<3 { features = (features - 1) / 2 + 1 }
            return features
        }
    }

    public let maxLength: Int
    public let imageTextLength: Int
    public let audioTextLength: Int
    public let headLayers: Int
    public let projectorHiddenSize: Int
    public let temperatures: [String: Double]
    public let bosTokenID: Int
    public let padTokenID: Int
    public let text: Text
    public let vision: Vision
    public let audio: Audio

    enum CodingKeys: String, CodingKey {
        case maxLength = "max_length"
        case imageTextLength = "image_text_length"
        case audioTextLength = "audio_text_length"
        case headLayers = "head_layers"
        case projectorHiddenSize = "projector_hidden_size"
        case temperatures
        case bosTokenID = "bos_token_id"
        case padTokenID = "pad_token_id"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
    }

    /// Reads `config.json` from a local model folder.
    public init(modelFolder: URL) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        self = try JSONDecoder().decode(D1Configuration.self, from: data)
    }
}
