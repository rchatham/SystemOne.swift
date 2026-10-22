import Foundation

/// librosa's `filters.mel(sr, n_fft, n_mels, norm="slaney")`, computed in
/// double precision the way NumPy does and stored as float32.
package enum MelFilterbank {
    /// `[melCount][fftSize / 2 + 1]` triangular filters with Slaney area normalization.
    package static func slaney(sampleRate: Int, fftSize: Int, melCount: Int) -> [[Float]] {
        let linearStep = 200.0 / 3
        let logStartHz = 1_000.0
        let logStep = log(6.4) / 27
        let logStartMel = logStartHz / linearStep

        func hertzToMel(_ hertz: Double) -> Double {
            hertz >= logStartHz ? logStartMel + log(hertz / logStartHz) / logStep : hertz / linearStep
        }
        func melToHertz(_ mel: Double) -> Double {
            mel >= logStartMel ? logStartHz * exp(logStep * (mel - logStartMel)) : linearStep * mel
        }

        // np.linspace: start + i * step, with the endpoint set exactly.
        let (low, high) = (hertzToMel(0), hertzToMel(Double(sampleRate) / 2))
        let count = melCount + 2
        let step = (high - low) / Double(count - 1)
        var mels = (0..<count).map { low + Double($0) * step }
        mels[count - 1] = high
        let edges = mels.map(melToHertz)

        let binWidth = 1.0 / (Double(fftSize) * (1.0 / Double(sampleRate)))
        let frequencies = (0...(fftSize / 2)).map { Double($0) * binWidth }
        return (0..<melCount).map { index in
            let lowerWidth = edges[index + 1] - edges[index]
            let upperWidth = edges[index + 2] - edges[index + 1]
            let enorm = 2.0 / (edges[index + 2] - edges[index])
            return frequencies.map { frequency in
                let rising = -(edges[index] - frequency) / lowerWidth
                let falling = (edges[index + 2] - frequency) / upperWidth
                let weight = Float(max(0, min(rising, falling)))
                return Float(Double(weight) * enorm)
            }
        }
    }
}
