import Foundation

/// Python string semantics that reference implementations depend on.
package enum PythonSemantics {
    /// CPython `len(str)` counts Unicode scalar values, not Swift grapheme clusters.
    package static func characterCount(_ text: String) -> Int {
        text.unicodeScalars.count
    }

    /// Matches CPython 3.12 `str.strip()` whitespace behavior.
    package static func pythonStrip(_ text: String) -> String {
        let scalars = text.unicodeScalars
        var lowerBound = scalars.startIndex
        var upperBound = scalars.endIndex

        while lowerBound < upperBound, isPythonWhitespace(scalars[lowerBound]) {
            scalars.formIndex(after: &lowerBound)
        }
        while lowerBound < upperBound {
            let previous = scalars.index(before: upperBound)
            guard isPythonWhitespace(scalars[previous]) else { break }
            upperBound = previous
        }
        return String(scalars[lowerBound..<upperBound])
    }

    private static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0009...0x000D,
             0x001C...0x0020,
             0x0085,
             0x00A0,
             0x1680,
             0x2000...0x200A,
             0x2028...0x2029,
             0x202F,
             0x205F,
             0x3000:
            return true
        default:
            return false
        }
    }
}
