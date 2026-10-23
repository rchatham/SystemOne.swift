import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(HarnessModel.self) private var model
    @State private var isImporting = false
    @State private var isTakingPhoto = false
    @State private var pickedPhoto: PhotosPickerItem?

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            Form {
                Section("Model") {
                    LabeledContent("Folder", value: HarnessModel.modelFolderName)
                    LabeledContent("Present", value: model.missingModelFiles.isEmpty ? "Yes" : "No")
                    Button("Import folder from Files…") { isImporting = true }
                        .disabled(isBusy)
                    if !model.missingModelFiles.isEmpty {
                        Text("Pick a LiquidAI/d1-omni-600M snapshot folder, or copy it into this app's folder in Finder (iPhone › Files) as \(HarnessModel.modelFolderName). Missing: \(model.missingModelFiles.joined(separator: ", ")).")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Engine") {
                    LabeledContent("Status", value: statusText)
                    if let loaded = model.loadedPrecision {
                        LabeledContent("Precision", value: loaded.rawValue)
                    } else {
                        Picker("Precision", selection: $model.precision) {
                            ForEach(HarnessPrecision.allCases, id: \.self) { precision in
                                Text(precision.label).tag(precision)
                            }
                        }
                        .disabled(isBusy)
                        if model.precision == .automatic {
                            Text("Loads float32 with at least \(megabytes(HarnessModel.float32RequiredBytes)) available, else float16 (half the memory, same top answers).")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    if case .failed(let message) = model.phase {
                        Text(message).font(.footnote).foregroundStyle(.red)
                    }
                    if model.phase == .ready || model.phase == .running {
                        Button("Unload", role: .destructive) { model.unloadEngine() }
                            .disabled(model.phase == .running)
                    } else {
                        Button("Load engine") { Task { await model.loadEngine() } }
                            .disabled(!model.missingModelFiles.isEmpty || isBusy)
                    }
                }

                Section("Media") {
                    PhotosPicker(selection: $pickedPhoto, matching: .images) {
                        Label("Choose photo", systemImage: "photo.on.rectangle")
                    }
                    if CameraPicker.isAvailable {
                        Button { isTakingPhoto = true } label: { Label("Take photo", systemImage: "camera") }
                    }
                    Button {
                        Task { await model.toggleRecording() }
                    } label: {
                        model.recorder.isRecording
                            ? Label("Stop recording", systemImage: "stop.circle.fill")
                            : Label("Record audio (up to 30 s)", systemImage: "mic")
                    }
                    .tint(model.recorder.isRecording ? .red : nil)
                    if let media = model.media {
                        if case .image(let preview, _) = media {
                            Image(uiImage: preview)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 220)
                        }
                        LabeledContent("Attached", value: media.summary)
                        Button("Remove", role: .destructive) { model.clearMedia() }
                    }
                }

                Section("Request") {
                    Picker("Questions", selection: $model.sample) {
                        ForEach(HarnessSample.allCases, id: \.self) { sample in
                            Text(sample.label).tag(sample)
                        }
                    }
                    TextField("State", text: $model.state, axis: .vertical)
                        .lineLimit(2...8)
                    TextField("Extra yes/no question (optional)", text: $model.customQuestion, axis: .vertical)
                    Picker("Batching", selection: $model.batching) {
                        ForEach(HarnessBatching.allCases, id: \.self) { batching in
                            Text(batching.label).tag(batching)
                        }
                    }
                    Button("Run systemOne") { Task { await model.run() } }
                        .disabled(model.phase != .ready || model.recorder.isRecording)
                    ForEach(model.answerLines, id: \.self) { line in
                        Text(line).font(.callout.monospaced())
                    }
                    if let usage = model.usage {
                        Text(usage).font(.footnote).foregroundStyle(.secondary)
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
                    // In MB like the threshold above and the self-test log.
                    LabeledContent("Available to process", value: megabytes(model.memory.availableToProcess))
                    LabeledContent("MLX active", value: bytes(model.memory.mlxActive))
                    LabeledContent("MLX cache", value: bytes(model.memory.mlxCache))
                    LabeledContent("MLX peak", value: bytes(model.memory.mlxPeak))
                    Button("Refresh") { model.refreshMemory() }
                }
            }
            .navigationTitle("System One")
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result {
                    Task { await model.importModel(from: url) }
                }
            }
            .sheet(isPresented: $isTakingPhoto) {
                CameraPicker { photo in Task { await model.attachPhoto(photo) } }
                    .ignoresSafeArea()
            }
            .onChange(of: pickedPhoto) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self), let photo = UIImage(data: data) {
                        await model.attachPhoto(photo)
                    }
                    pickedPhoto = nil
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
        case .missingModel: "No model"
        case .idle: "Not loaded"
        case .importing: "Importing…"
        case .loading: "Verifying and loading…"
        case .ready: "Ready"
        case .running: "Running…"
        case .failed: "Failed"
        }
    }

    /// Whole MiB, the unit the self-test logs and the README use.
    private func megabytes(_ count: Int) -> String {
        "\((count / 1_048_576).formatted()) MB"
    }

    private func bytes(_ count: Int) -> String {
        Int64(count).formatted(.byteCount(style: .memory))
    }
}
