import Foundation
import MLX

/// The gated delta rule (Gated DeltaNet), evaluated in chunks as transformers'
/// `torch_chunk_gated_delta_rule` does.
///
/// Per head, with state `S` (`[keyDim, valueDim]`), token `t` applies
/// `S ← exp(g_t)·S`, then `S ← S + β_t·k_t ⊗ (v_t − Sᵀk_t)`, and reads
/// `o_t = Sᵀq_t`. Within a chunk the updates are condensed into matrix products
/// through the UT transform, a unit lower-triangular system solved by forward
/// substitution; only the chunks themselves run in sequence.
enum GatedDeltaRule {
    /// Runs the recurrence over `[batch, tokens, heads, …]` inputs in float32.
    ///
    /// - Parameters:
    ///   - query: `[B, T, H, Dk]`, L2-normalized and scaled by `1/√Dk`.
    ///   - key: `[B, T, H, Dk]`, L2-normalized.
    ///   - value: `[B, T, H, Dv]`.
    ///   - decay: `[B, T, H]`, the log of each step's decay (`≤ 0`).
    ///   - beta: `[B, T, H]`, each step's write strength.
    ///   - state: `[B, H, Dk, Dv]`, or `nil` for zeros.
    /// - Returns: outputs `[B, T, H, Dv]` and the final state.
    static func run(
        query: MLXArray, key: MLXArray, value: MLXArray, decay: MLXArray, beta: MLXArray,
        state: MLXArray?, chunkSize: Int = 64
    ) -> (output: MLXArray, state: MLXArray) {
        let (batch, length, heads, keyWidth) = (key.dim(0), key.dim(1), key.dim(2), key.dim(3))
        let valueWidth = value.dim(3)
        let chunk = min(chunkSize, length)
        let padding = (chunk - length % chunk) % chunk
        let chunks = (length + padding) / chunk

        // [B, H, T, …] in float32, padded to whole chunks.
        func perHead(_ x: MLXArray) -> MLXArray {
            var x = x.asType(.float32).transposed(0, 2, 1, 3)
            if padding > 0 { x = concatenated([x, MLXArray.zeros([batch, heads, padding, x.dim(3)])], axis: 2) }
            return x
        }
        func steps(_ x: MLXArray) -> MLXArray {
            var x = x.asType(.float32).transposed(0, 2, 1)
            if padding > 0 { x = concatenated([x, MLXArray.zeros([batch, heads, padding])], axis: 2) }
            return x
        }
        let q = perHead(query), k = perHead(key), v = perHead(value)
        let b = steps(beta)
        let g = steps(decay).reshaped(batch, heads, chunks, chunk)

        let keyBeta = (k * b.expandedDimensions(axis: -1)).reshaped(batch, heads, chunks, chunk, keyWidth)
        let valueBeta = (v * b.expandedDimensions(axis: -1)).reshaped(batch, heads, chunks, chunk, valueWidth)
        let queries = q.reshaped(batch, heads, chunks, chunk, keyWidth)
        let keys = k.reshaped(batch, heads, chunks, chunk, keyWidth)

        // cumulative[t] = log of the decay from the chunk's start through t;
        // pairwise[i, j] = exp(cumulative[i] − cumulative[j]) for j ≤ i, else 0.
        let cumulative = g.cumsum(axis: -1)
        let lower = MLXArray((0..<chunk).flatMap { i in (0..<chunk).map { j in j <= i } }, [chunk, chunk])
        let difference = cumulative.expandedDimensions(axis: -1) - cumulative.expandedDimensions(axis: -2)
        let pairwise = MLX.exp(MLX.where(lower, difference, MLXArray(-Float.infinity)))

        let keysT = keys.swappedAxes(-1, -2)
        let system = matmul(keyBeta, keysT) * pairwise
        let intra = matmul(queries, keysT) * pairwise
        let decayedKeyBeta = keyBeta * MLX.exp(cumulative).expandedDimensions(axis: -1)

        // Invert I + strictly-lower(system) by forward substitution: row i of
        // T = −strictly-lower(system) gains T[i, :i] · T[:i, :i]. Row i is zero
        // from column i on, so the product reads only finished rows.
        var inverse = -tril(system, k: -1)
        for row in 1..<chunk {
            let current = inverse[.ellipsis, row ..< (row + 1), 0...]
            inverse[.ellipsis, row ..< (row + 1), 0...] = current + matmul(current, inverse)
        }
        inverse = inverse + eye(chunk)
        let newValues = matmul(inverse, valueBeta)
        let keyCumulativeDecay = matmul(inverse, decayedKeyBeta)

        let decayedQueries = queries * MLX.exp(cumulative).expandedDimensions(axis: -1)
        let last = cumulative[.ellipsis, (chunk - 1) ..< chunk]
        let decayedKeysT = (keys * MLX.exp(last - cumulative).expandedDimensions(axis: -1)).swappedAxes(-1, -2)
        let chunkDecay = MLX.exp(last).expandedDimensions(axis: -1)

        var current = state?.asType(.float32) ?? MLXArray.zeros([batch, heads, keyWidth, valueWidth])
        var outputs: [MLXArray] = []
        for index in 0..<chunks {
            let fresh = newValues[0..., 0..., index] - matmul(keyCumulativeDecay[0..., 0..., index], current)
            outputs.append(matmul(decayedQueries[0..., 0..., index], current) + matmul(intra[0..., 0..., index], fresh))
            current = current * chunkDecay[0..., 0..., index] + matmul(decayedKeysT[0..., 0..., index], fresh)
        }
        let output = stacked(outputs, axis: 2)
            .reshaped(batch, heads, chunks * chunk, valueWidth)[0..., 0..., 0 ..< length]
            .transposed(0, 2, 1, 3)
        return (output, current)
    }

    /// `x / √(Σx² + ε)` over the last axis, as FLA's `l2norm` (ε = 1e-6).
    static func l2Normalized(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
        x * (x.square().sum(axis: -1, keepDims: true) + epsilon).rsqrt()
    }
}
