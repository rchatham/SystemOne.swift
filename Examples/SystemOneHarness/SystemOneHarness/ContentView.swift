import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(HarnessModel.self) private var model
    @State private var isImporting = false

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            Form {
                Section("Weights") {
                    LabeledContent("File", value: HarnessModel.weightsFileName)
                    LabeledContent("Present", value: model.weightsArePresent ? "Yes" : "No")
                    Button("Import from Files…") { isImporting = true }
                        .disabled(isBusy)
                    if !model.weightsArePresent {
                        Text("Or copy the file into this app's folder in Finder (iPhone › Files).")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Engine") {
                    LabeledContent("Status", value: statusText)
                    if case .failed(let message) = model.phase {
                        Text(message).font(.footnote).foregroundStyle(.red)
                    }
                    if model.phase == .ready || model.phase == .running {
                        Button("Unload", role: .destructive) { model.unloadEngine() }
                            .disabled(model.phase == .running)
                    } else {
                        Button("Load engine") { Task { await model.loadEngine() } }
                            .disabled(!model.weightsArePresent || isBusy)
                    }
                }

                Section("Sample request") {
                    TextField("State", text: $model.state, axis: .vertical)
                        .lineLimit(3...8)
                    Picker("Batching", selection: $model.batching) {
                        ForEach(HarnessBatching.allCases, id: \.self) { batching in
                            Text(batching.label).tag(batching)
                        }
                    }
                    Button("Run systemOne") { Task { await model.runSample() } }
                        .disabled(model.phase != .ready)
                    ForEach(model.answerLines, id: \.self) { line in
                        Text(line).font(.callout.monospaced())
                    }
                }

                Section("Timings") {
                    ForEach(Array(model.measurements.enumerated()), id: \.offset) { _, measurement in
                        LabeledContent(
                            measurement.label,
                            value: "\(measurement.milliseconds.formatted(.number.precision(.fractionLength(1)))) ms"
                        )
                    }
                }

                Section("Memory") {
                    LabeledContent("Available to process", value: bytes(model.memory.availableToProcess))
                    LabeledContent("MLX active", value: bytes(model.memory.mlxActive))
                    LabeledContent("MLX cache", value: bytes(model.memory.mlxCache))
                    LabeledContent("MLX peak", value: bytes(model.memory.mlxPeak))
                    Button("Refresh") { model.refreshMemory() }
                }
            }
            .navigationTitle("Von Harness")
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.data]) { result in
                if case .success(let url) = result {
                    Task { await model.importWeights(from: url) }
                }
            }
        }
    }

    private var isBusy: Bool {
        switch model.phase {
        case .importing, .loading, .running: true
        default: false
        }
    }

    private var statusText: String {
        switch model.phase {
        case .missingWeights: "No weights"
        case .idle: "Not loaded"
        case .importing: "Importing…"
        case .loading: "Verifying and loading…"
        case .ready: "Ready"
        case .running: "Running…"
        case .failed: "Failed"
        }
    }

    private func bytes(_ count: Int) -> String {
        Int64(count).formatted(.byteCount(style: .memory))
    }
}
