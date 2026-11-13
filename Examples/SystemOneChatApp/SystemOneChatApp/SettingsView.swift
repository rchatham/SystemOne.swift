import SwiftUI
import SystemOneD1

struct SettingsView: View {
    @Environment(ChatModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var choosingFolder = false

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            Form {
                Section {
                    if RouterKind.supportsOnDevice {
                        Picker("Router", selection: $model.routerKind) {
                            ForEach(RouterKind.allCases) { Text($0.label).tag($0) }
                        }
                    } else {
                        LabeledContent("Router", value: RouterKind.server.label)
                    }
                } footer: {
                    if !RouterKind.supportsOnDevice {
                        Text("MLX cannot run in the iOS Simulator, so routing goes to system-one serve on this Mac.")
                    }
                }

                switch model.routerKind {
                case .server:
                    Section {
                        TextField("Endpoint", text: $model.serverURL)
                            .autocorrectionDisabled()
                            #if os(iOS)
                            .textInputAutocapitalization(.never)
                            .keyboardType(.URL)
                            #endif
                    } header: {
                        Text("Server")
                    } footer: {
                        Text("Start it with: swift run system-one serve --port 8083 --model-folder <d1-omni-600M>")
                            .textSelection(.enabled)
                    }
                case .onDevice:
                    Section("d1-omni-600M") {
                        LabeledContent("Folder", value: model.modelFolder?.lastPathComponent ?? "Not chosen")
                        Button("Choose Folder…") { choosingFolder = true }
                        Picker("Precision", selection: $model.precision) {
                            Text("float32").tag(D1Precision.float32)
                            Text("float16").tag(D1Precision.float16)
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Restart") {
                        dismiss()
                        Task { await model.start() }
                    }
                }
            }
            .fileImporter(isPresented: $choosingFolder, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { model.chooseModelFolder(url) }
            }
        }
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 300)
        #endif
    }
}
