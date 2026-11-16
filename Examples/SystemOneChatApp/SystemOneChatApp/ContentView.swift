import SwiftUI
import SystemOneChat

struct ContentView: View {
    @Environment(ChatModel.self) private var model
    @State private var showingSettings = false
    @FocusState private var composerFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                StatusBanner(showSettings: { showingSettings = true })
                transcript
                Divider()
                composer
            }
            .navigationTitle("System One Chat")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Settings", systemImage: "gearshape") { showingSettings = true }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
            }
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(model.entries) { entry in
                        MessageRow(entry: entry).id(entry.id)
                    }
                    if model.phase == .sending {
                        ProgressView().padding(.leading, 4).id("thinking")
                    }
                }
                .padding()
            }
            .onChange(of: model.entries.count) {
                withAnimation { proxy.scrollTo(model.entries.last?.id, anchor: .bottom) }
            }
        }
    }

    private var composer: some View {
        @Bindable var model = model
        return HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $model.draft, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.roundedBorder)
                .focused($composerFocused)
                .onSubmit(send)
                .disabled(model.phase != .ready)
            Button("Send", systemImage: "arrow.up.circle.fill", action: send)
                .labelStyle(.iconOnly)
                .font(.title2)
                .disabled(!model.canSend)
                .keyboardShortcut(.return, modifiers: .command)
        }
        .padding()
    }

    private func send() {
        Task {
            await model.send()
            composerFocused = true
        }
    }
}

/// Shows startup progress and failures above the transcript.
private struct StatusBanner: View {
    @Environment(ChatModel.self) private var model
    let showSettings: () -> Void

    var body: some View {
        switch model.phase {
        case .starting:
            Label {
                Text(model.routerKind == .onDevice ? "Loading d1…" : "Connecting to system-one serve…")
            } icon: {
                ProgressView().controlSize(.small)
            }
            .bannerStyle(.secondary)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                Text(message).foregroundStyle(.red)
                HStack {
                    Button("Settings", action: showSettings)
                    Button("Retry") { Task { await model.start() } }
                }
            }
            .bannerStyle(.red)
        case .stopped:
            EmptyView()
        case .ready, .sending:
            if let summary = model.routerSummary {
                Text("Routing: \(summary) · Replies: Apple Intelligence")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                    .padding(.top, 8)
            }
        }
    }
}

private extension View {
    func bannerStyle(_ tint: Color) -> some View {
        frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(tint.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
            .padding([.horizontal, .top])
    }
}

private struct MessageRow: View {
    let entry: ChatEntry

    var body: some View {
        switch entry.role {
        case .user:
            Text(entry.text)
                .textSelection(.enabled)
                .padding(10)
                .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
                .frame(maxWidth: .infinity, alignment: .trailing)
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                RouteTrace(calls: entry.calls)
                Text(entry.text).textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .error:
            Label(entry.text, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// The router's choices for one reply, e.g. "save_note 0.97 → generate_text".
private struct RouteTrace: View {
    let calls: [ToolCall]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(calls.map(Self.describe).joined(separator: " → "))
            ForEach(Array(calls.dropLast().enumerated()), id: \.offset) { _, call in
                Text(call.observation).lineLimit(3)
            }
        }
        .font(.caption.monospaced())
        .foregroundStyle(.secondary)
    }

    private static func describe(_ call: ToolCall) -> String {
        guard let confidence = call.confidence else { return call.tool }
        return "\(call.tool) \(confidence.formatted(.number.precision(.fractionLength(2))))"
    }
}
