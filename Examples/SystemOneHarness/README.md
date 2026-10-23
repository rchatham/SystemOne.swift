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
- About 3.4 GB of memory at peak in float32 (2.24 GB of weights plus
  activations), or 1.7 GB in float16. **Precision › Automatic** loads float32
  when `os_proc_available_memory()` reports at least 3.6 GB and float16
  otherwise, so 4 GB devices such as the iPhone 13 get float16. Verified on an
  iPhone 14 Pro (6 GB) in both precisions.
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
5. Pick a precision (Automatic, float32, or float16) and tap **Load engine**.
   This hashes the 2.35 GB weights against the pinned SHA-256 and copies every
   tensor into MLX (about 7 s on an iPhone 14 Pro). The Engine section then
   shows the precision that was loaded.
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

Add `-precision float32` or `-precision float16` after `-selfTest` to override
the automatic choice. On an iPhone 14 Pro (Release, 4.07 GB available before
load, warm Metal kernels):

| | float32 | float16 |
|---|---|---|
| Load (verify + copy) | 7.3 s | 6.7 s |
| Text, three questions | 0.35 s | 0.27 s |
| Photo, three questions | 5.0 s | 3.8 s |
| 3 s voice clip | 0.32 s | 0.21 s |
| MLX active / peak | 2.24 / 3.36 GB | 1.12 / 1.68 GB |
| Available after the run | 1.7 GB | 2.8 GB |

Both precisions give the same top answers; probabilities differ by under a
percentage point. The first float16 launch after installing is slower
(text 1.7 s, voice 0.9 s) while Metal compiles its half-precision kernels.
A photo is slow because every question re-runs the trunk over the roughly
1,800-position image prefix, as the reference does.

## Memory

The engine keeps its weights in MLX arrays for its whole lifetime: 2.24 GB in
float32, 1.12 GB in float16. float16 weights are converted on the CPU while
loading, so MLX never holds a float32 copy. The harness caps MLX's buffer cache at 64 MB, and the Memory section
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
