import SwiftUI

@main
struct SystemOneHarnessApp: App {
    @State private var model = HarnessModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .task {
                    if ProcessInfo.processInfo.arguments.contains("-selfTest") {
                        await model.selfTest()
                    }
                }
        }
        .onChange(of: scenePhase) { _, newPhase in
            // Files copied in through Finder appear while the app is in the background.
            if newPhase == .active {
                model.refreshPhase()
            }
        }
    }
}
