# SystemOneHarness

A SwiftUI app for running d1-omni-600M through `SystemOneEngine` on an iPhone or
iPad. It answers text, photo, and voice requests on device: choose a photo,
take one with the camera, or record up to 30 seconds from the microphone, then
run a built-in question set (with an optional extra yes/no question) and see
answers, timings, and process/MLX memory.

## Requirements

- Xcode 16 or later
- A **physical** Apple-silicon iOS device on iOS 17 or later. MLX needs Metal
  GPU features the iOS Simulator does not provide, so inference cannot run
  there. For development on a Mac, pick the **My Mac (Designed for iPad)**
  destination.
- About 3.4 GB of memory for the app at peak (2.24 GB of float32 weights plus
  activations). Verified on an iPhone 14 Pro (6 GB); 4 GB devices such as the
  iPhone 13 are not expected to fit.
- A paid Apple Developer team for signing. The app requests the
  `com.apple.developer.kernel.increased-memory-limit` entitlement, which
  automatic signing adds to the provisioning profile.
- The [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
  snapshot (`config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `model.safetensors`), LFM Open License v1.0.

## Run

1. Open `Examples/SystemOneHarness/SystemOneHarness.xcodeproj`. Xcode resolves the local
   `SystemOne` package from the repository root.
2. Select the **SystemOneHarness** target, then under *Signing & Capabilities* set
   your team and a unique bundle identifier (the placeholder is
   `com.example.SystemOneHarness`).
3. Run on your device. The shared scheme runs the **Release** configuration;
   MLX is far slower when built Debug.
4. Put the snapshot in the app's `Documents/d1-omni-600M` folder, any of these ways:
   - Finder: select the device, open the **Files** tab, and drag a folder named
     `d1-omni-600M` onto **System One**.
   - In the app: **Import folder from Files…** copies the four files from a
     folder you have already placed in the Files app.
   - From a Mac:
     `xcrun devicectl device copy to --device <id> --domain-type appDataContainer --domain-identifier <bundle id> --source model.safetensors --destination Documents/d1-omni-600M/model.safetensors`
     (repeat for each file).
5. Tap **Load engine**. This hashes the 2.35 GB weights against the pinned
   SHA-256 and copies every tensor into MLX (about 7 s on an iPhone 14 Pro).
6. Attach a photo or a recording if you like, pick a question set, and tap
   **Run systemOne**. iOS asks for camera or microphone access the first time.

Photos are rendered upright (EXIF orientation applied) in sRGB. Recordings are
16-bit mono 16 kHz PCM, the model's input format; other audio is resampled.

## Self-test

Launching with `-selfTest` loads the model and runs a text request, a synthetic
photo (decoded through the same `UIImage` path as the picker), and synthetic
16 kHz and 44.1 kHz stereo WAV files (through the same reader as the recorder),
printing answers, timings, and memory:

```bash
xcrun devicectl device process launch --device <id> --console --terminate-existing <bundle id> -- -selfTest
```

On an iPhone 14 Pro (Release): load 6.7 s; text 0.35 s; a three-question photo
request 2.9 s; a 3 s voice clip 0.25–0.29 s; MLX peak 3.35 GB with 1.7 GB still
available. The photo's roughly 1,800-position image prefix runs through the
trunk once per request and every question reuses its states (5.0 s when each
question re-ran it, as the reference does).

## Memory

The engine keeps the float32 weights in MLX arrays (2.24 GB) for its whole
lifetime. The harness caps MLX's buffer cache at 64 MB, and the Memory section
shows `os_proc_available_memory()`: the headroom before iOS terminates the app.
If the app is killed while loading, check the device's jetsam reports in
*Settings › Privacy & Security › Analytics & Improvements*. Inference only runs
in the foreground; iOS does not allow GPU work from the background.

## Command-line build

```bash
xcodebuild build \
  -project Examples/SystemOneHarness/SystemOneHarness.xcodeproj \
  -scheme SystemOneHarness \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO
```

CI runs this build on every push, which confirms that `SystemOne`, `SystemOneD1`, and MLX
compile for iOS. It cannot run inference.
