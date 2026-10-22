# SystemOneHarness

A minimal SwiftUI app for running `SystemOneEngine` on an iPhone or iPad. It loads
the verified production weights, runs a fixed four-question `systemOne`
request (Choice, default Noul, explicit Noul, Score), and shows answers, load
and inference timings, and process/MLX memory.

## Requirements

- Xcode 16 or later
- A **physical** Apple-silicon iOS device on iOS 17 or later. MLX needs Metal
  GPU features the iOS Simulator does not provide, so inference cannot run
  there. For development on a Mac, pick the **My Mac (Designed for iPad)**
  destination.
- About 1.6 GB of free memory for the app. Plan on a device with 8 GB of RAM
  (iPhone 15 Pro or later).
- A paid Apple Developer team for signing. The app requests the
  `com.apple.developer.kernel.increased-memory-limit` entitlement, which
  automatic signing adds to the provisioning profile.
- The verified `option_marker.safetensors` artifact, produced on a Mac with
  `tools/download_model.py` followed by `tools/convert_option_marker.py`.

## Run

1. Open `Examples/SystemOneHarness/SystemOneHarness.xcodeproj`. Xcode resolves the local
   `SystemOne` package from the repository root.
2. Select the **SystemOneHarness** target, then under *Signing & Capabilities* set
   your team and a unique bundle identifier (the placeholder is
   `com.example.SystemOneHarness`).
3. Run on your device. The shared scheme runs the **Release** configuration;
   MLX is far slower when built Debug.
4. Get the weights into the app's Documents folder, either way works:
   - Finder: select the device, open the **Files** tab, and drag
     `option_marker.safetensors` onto **Von Harness**.
   - In the app: **Import from Files…** copies a file you have already placed
     in the Files app.
5. Tap **Load engine**. This hashes the full 1.58 GB file against the pinned
   SHA-256 and copies every tensor into MLX, so expect several seconds.
6. Tap **Run systemOne**. The first run includes Metal kernel compilation;
   run it a few times for a steady latency.

The tokenizer is bundled directly from
`Tests/SystemOneTests/Resources/Tokenizer`, so it always matches the pinned
revision the tests use.

## Memory

The engine keeps the float32 weights in MLX arrays (about 1.58 GB) for its
whole lifetime. The harness caps MLX's buffer cache at 64 MB, and the Memory
section shows `os_proc_available_memory()`: the headroom before iOS
terminates the app. If the app is killed while loading, check the device's
jetsam reports in *Settings › Privacy & Security › Analytics & Improvements*.
Inference only runs in the foreground; iOS does not allow GPU work from the
background.

## Command-line build

```bash
xcodebuild build \
  -project Examples/SystemOneHarness/SystemOneHarness.xcodeproj \
  -scheme SystemOneHarness \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO
```

CI runs this build on every push, which confirms that `SystemOne`, MLX, and the
tokenizer compile for iOS. It cannot run inference.
