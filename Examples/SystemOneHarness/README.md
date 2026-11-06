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
- Up to about 3.6 GB of MLX memory in float32 (2.24 GB of weights plus
  activations for the largest image), or 1.8 GB in float16.
  **Precision › Automatic** loads float32 when `os_proc_available_memory()`
  reports at least 3,900 MB and float16 otherwise (an iOS app on a Mac, where it
  reports 0, keeps float32). Verified on an iPhone 14 Pro (6 GB, 4,074 MB
  available) in both precisions, and on a 4 GB iPhone 13 (2,331 MB available),
  where Automatic loads float16 and the largest photo leaves 1,071 MB available.
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
the automatic choice. On an iPhone 14 Pro (Release, 4,074 MB available before
load, warm Metal kernels):

| | float32 | float16 |
|---|---|---|
| Load (verify + copy) | 8.2–9.8 s | 5.5–7.0 s |
| Text, three questions | 0.37 s | 0.25 s |
| Photo 1200×900, three questions | 2.9 s | 2.1 s |
| 3 s voice clip | 0.32 s | 0.19 s |
| Photo 2560×1024 (largest prefix), three questions | 4.5 s | 3.2 s |
| MLX active | 2.24 GB | 1.12 GB |
| MLX peak: 1200×900 / 2560×1024 photo | 3.35 / 3.60 GB | 1.69 / 1.80 GB |
| Available after the run | 1.7 GB | 2.75 GB |

Both precisions give the same top answers; probabilities differ by under a
percentage point. The first launch after installing is slower while Metal
compiles its kernels (float32 text 2.6 s and voice 1.5 s; float16 text 1.4 s
and voice 0.8 s). On a 4 GB iPhone 13 (Release, 2,331 MB available before
load), Automatic loaded float16 and every request passed with the same answers
as the 14 Pro. MLX peaks were the same as the 14 Pro's float16 run (1.69 GB
for the 1200×900 photo, 1.78 GB for 2560×1024), with 1,071 MB still available
at the end. Its timings so far come from a first launch before the prefix
cache below, which includes compiling the Metal kernels: load 21 s, and 5.8 s
for the 1200×900 photo once warm.

The self-test prints each run's own MLX peak; the 2560×1024 photo (2,810
prefix positions, near the 2,816 maximum) sets the 3,900 MB float32 threshold.
A photo request runs the trunk once over the image prefix (about 1,800
positions for 1200×900) and caches its states; each question then runs only
its own text positions against them. Re-running the prefix for every question,
as the reference does, took 5.0 s / 3.8 s (float32 / float16) for the
1200×900 photo and 8.4 s / 6.0 s for 2560×1024. The peak is set by building
the prefix, so the cache does not lower it.

If `devicectl` hangs or reports a tunnel failure, disconnect any VPN on the
Mac: it can block the network tunnel that CoreDevice opens to the device.

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
