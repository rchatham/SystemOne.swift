import SwiftUI

@main
struct SystemOneChatApp: App {
    @State private var model = ChatModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .task { await model.start() }
        }
        #if os(macOS)
        .defaultSize(width: 520, height: 680)
        #endif
    }
}
