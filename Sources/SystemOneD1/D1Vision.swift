// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX
import MLXNN
import SystemOne

/// `vision.py`: LFM2-VL tiling, a SigLIP2 NaFlex tower, and a 2×2
/// pixel-unshuffle projector into the trunk's width.
final class D1Vision {
    static let tile = 512
    static let patch = 16

    /// `layout`: LFM2-VL's smart resize, tile grid, and thumbnail.
    struct Layout: Equatable {
        /// Columns and rows of 512-pixel tiles; `(1, 1)` when not tiled.
        let grid: (columns: Int, rows: Int)
        let thumbnail: (height: Int, width: Int)
        let tiled: Bool

        static func == (lhs: Layout, rhs: Layout) -> Bool {
            lhs.grid == rhs.grid && lhs.thumbnail == rhs.thumbnail && lhs.tiled == rhs.tiled
        }

        /// Crop sizes in encoding order: tiles row by row, then the thumbnail.
        var crops: [(height: Int, width: Int)] {
            let tiles = tiled ? Array(repeating: (height: tile, width: tile), count: grid.columns * grid.rows) : []
            return tiles + [thumbnail]
        }
    }

    /// Tile grids `(columns, rows)` with 2 to 10 tiles, in the exact order
    /// CPython's `sorted(set(...), key=area)` visits them in `vision.py`.
    static let tileGrids: [(Int, Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3),
        (7, 1), (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10),
    ]

    static func layout(width: Int, height: Int) -> Layout {
        let factor = 32.0
        let maximum = 256.0 * 1_024
        let minimum = 64.0 * 1_024
        let (w0, h0) = (Double(width), Double(height))
        func roundEven(_ x: Double) -> Double { x.rounded(.toNearestOrEven) }
        var h = max(factor, roundEven(h0 / factor) * factor)
        var w = max(factor, roundEven(w0 / factor) * factor)
        if h * w > maximum {
            let beta = (h0 * w0 / maximum).squareRoot()
            h = max(factor, (h0 / beta / factor).rounded(.down) * factor)
            w = max(factor, (w0 / beta / factor).rounded(.down) * factor)
        } else if h * w < minimum {
            let beta = (minimum / (h0 * w0)).squareRoot()
            h = (h0 * beta / factor).rounded(.up) * factor
            w = (w0 * beta / factor).rounded(.up) * factor
        }
        let large = max(16, roundEven(h0 / factor) * factor) * max(16, roundEven(w0 / factor) * factor) > maximum * 2
        var grid = (1, 1)
        if large {
            var best = Double.infinity
            for ratio in tileGrids {
                let difference = abs(w0 / h0 - Double(ratio.0) / Double(ratio.1))
                if difference < best
                    || (difference == best && w0 * h0 > 0.5 * Double(tile * tile * ratio.0 * ratio.1)) {
                    grid = ratio
                    best = difference
                }
            }
        }
        return Layout(grid: (grid.0, grid.1), thumbnail: (Int(h), Int(w)), tiled: large)
    }

    /// Prefix embeddings an image occupies: one per 2×2 block of 16-pixel patches.
    static func prefixLength(width: Int, height: Int) -> Int {
        layout(width: width, height: height).crops.reduce(0) { $0 + ($1.height / patch) * ($1.width / patch) / 4 }
    }

    private struct Layer {
        let norm1Weight: MLXArray, norm1Bias: MLXArray
        let query: Linear, key: Linear, value: Linear, output: Linear
        let norm2Weight: MLXArray, norm2Bias: MLXArray
        let expand: Linear, contract: Linear
    }

    private let configuration: D1Configuration.Vision
    private let patchEmbedding: Linear
    /// `[side, side, hidden]` learned position grid.
    private let positionGrid: MLXArray
    private let layers: [Layer]
    private let postNormWeight: MLXArray
    private let postNormBias: MLXArray
    private let projectorExpand: Linear
    private let projectorContract: Linear

    init(weights: D1Weights) {
        configuration = weights.configuration.vision
        let tower = "vision.tower.vision_model"
        func linear(_ name: String) -> Linear {
            Linear(weight: weights.array("\(name).weight"), bias: weights.array("\(name).bias"))
        }
        patchEmbedding = linear("\(tower).embeddings.patch_embedding")
        let side = Int(Double(configuration.numPatches).squareRoot())
        positionGrid = weights.array("\(tower).embeddings.position_embedding.weight")
            .reshaped(side, side, configuration.hiddenSize)
        layers = (0..<configuration.numHiddenLayers).map { index in
            let prefix = "\(tower).encoder.layers.\(index)"
            return Layer(
                norm1Weight: weights.array("\(prefix).layer_norm1.weight"),
                norm1Bias: weights.array("\(prefix).layer_norm1.bias"),
                query: linear("\(prefix).self_attn.q_proj"),
                key: linear("\(prefix).self_attn.k_proj"),
                value: linear("\(prefix).self_attn.v_proj"),
                output: linear("\(prefix).self_attn.out_proj"),
                norm2Weight: weights.array("\(prefix).layer_norm2.weight"),
                norm2Bias: weights.array("\(prefix).layer_norm2.bias"),
                expand: linear("\(prefix).mlp.fc1"),
                contract: linear("\(prefix).mlp.fc2")
            )
        }
        postNormWeight = weights.array("\(tower).post_layernorm.weight")
        postNormBias = weights.array("\(tower).post_layernorm.bias")
        projectorExpand = linear("vision.projector.linear_1")
        projectorContract = linear("vision.projector.linear_2")
    }

    /// `preprocess`: every crop as interleaved RGB bytes, resized exactly as
    /// torchvision's antialiased bilinear resize does on uint8.
    static func crops(of image: RGBImage) -> [(height: Int, width: Int, pixels: [UInt8])] {
        let plan = layout(width: image.width, height: image.height)
        func resized(width: Int, height: Int) -> [UInt8] {
            AntialiasedResize.resize(image.pixels, width: image.width, height: image.height, channels: 3,
                                     toWidth: width, toHeight: height)
        }
        var crops: [(height: Int, width: Int, pixels: [UInt8])] = []
        if plan.tiled {
            let (columns, rows) = plan.grid
            let bigWidth = columns * tile
            let big = resized(width: bigWidth, height: rows * tile)
            for row in 0..<rows {
                for column in 0..<columns {
                    var pixels = [UInt8]()
                    pixels.reserveCapacity(tile * tile * 3)
                    for y in (row * tile)..<((row + 1) * tile) {
                        let start = (y * bigWidth + column * tile) * 3
                        pixels += big[start..<(start + tile * 3)]
                    }
                    crops.append((tile, tile, pixels))
                }
            }
        }
        let thumbnail = plan.thumbnail
        crops.append((thumbnail.height, thumbnail.width, resized(width: thumbnail.width, height: thumbnail.height)))
        return crops
    }

    /// `[P, D]` prefix embeddings for one image: every tile, then the thumbnail.
    func callAsFunction(_ image: RGBImage) -> MLXArray {
        concatenated(Self.crops(of: image).map { encode(crop: $0) }, axis: 0)
    }

    /// One crop through the tower and projector: `[(h/32)·(w/32), D]`.
    func encode(crop: (height: Int, width: Int, pixels: [UInt8])) -> MLXArray {
        let (rows, columns) = (crop.height / Self.patch, crop.width / Self.patch)
        return project(towerOutput(crop: crop).reshaped(1, rows, columns, configuration.hiddenSize))
    }

    /// SigLIP2's last hidden state for one crop: `[1, rows · columns, hidden]`.
    func towerOutput(crop: (height: Int, width: Int, pixels: [UInt8])) -> MLXArray {
        let (rows, columns) = (crop.height / Self.patch, crop.width / Self.patch)
        // RGB in [-1, 1], cut into 16×16 patches flattened as (dy, dx, channel).
        let normalized = (MLXArray(crop.pixels, [crop.height, crop.width, 3]).asType(.float32) - 127.5) / 127.5
        let patches = normalized
            .reshaped(rows, Self.patch, columns, Self.patch, 3)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(1, rows * columns, 3 * Self.patch * Self.patch)
        return tower(patchEmbedding(patches) + positions(rows: rows, columns: columns))
    }

    /// SigLIP2's position grid resized to the crop's patch grid, flattened row-major.
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
        // [H, W, C]: resample along W first, then H, as ATen's separable pass does.
        let alongWidth = matmul(matrix(side, columns), positionGrid)
        let alongHeight = matmul(matrix(side, rows), alongWidth.swappedAxes(0, 1)).swappedAxes(0, 1)
        return alongHeight.reshaped(1, rows * columns, configuration.hiddenSize)
    }

    /// The unpadded SigLIP2 encoder: pre-norm layers, then the post layer norm.
    private func tower(_ input: MLXArray) -> MLXArray {
        let eps = configuration.layerNormEps
        let heads = configuration.numAttentionHeads
        let headDimension = configuration.hiddenSize / heads
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

    /// LFM2-VL's 2×2 pixel unshuffle and two-layer GELU MLP.
    private func project(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let unshuffled = x
            .reshaped(batch, height, width / 2, channels * 2).transposed(0, 2, 1, 3)
            .reshaped(batch, width / 2, height / 2, channels * 4).transposed(0, 2, 1, 3)
        let projected = projectorContract(gelu(projectorExpand(unshuffled)))
        return projected.reshaped(-1, projected.dim(-1))
    }
}
