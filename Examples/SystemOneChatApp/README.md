# SystemOneChatApp

A SwiftUI chat app for the `SystemOneChat` agent on iPhone, iPad, and Mac. d1-omni
decides what each message asks for and which task runs; Apple Intelligence writes
every reply. Each reply shows the route d1 took, e.g. `save_note 0.97 → generate_text`.

One multiplatform target runs natively on macOS (not as an iPad app) and on iOS.

## Requirements

- Xcode 26 or later, with the Metal Toolchain:
  `xcodebuild -downloadComponent MetalToolchain`
- macOS 26 or iOS 26 with Apple Intelligence turned on. The iOS Simulator uses
  the Mac's model, so the Mac needs it on too.
- The [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
  snapshot (`config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `model.safetensors`), LFM Open License v1.0.

## Where routing runs

| Destination | Router | Notes |
|---|---|---|
| My Mac | d1 in the app (MLX) | Pick the snapshot folder in Settings |
| iPhone/iPad device | d1 in the app (MLX) | float16 by default; see the harness README for memory |
| iOS Simulator | `system-one serve` on the Mac | MLX needs Metal features the simulator lacks |

Every destination can also use `system-one serve`: open **Settings › Router**.

## Run on the Mac

1. Open `Examples/SystemOneChatApp/SystemOneChatApp.xcodeproj`. Xcode resolves the
   local `SystemOne` package from the repository root.
2. Choose the **SystemOneChatApp** scheme and the **My Mac** destination, then Run.
   The scheme runs **Release**, because MLX is far slower when built Debug. The
   Mac build signs to run locally, so no team is needed.
3. The first launch reports that no folder is chosen. Open **Settings**, click
   **Choose Folder…**, pick the d1 snapshot folder, and click **Restart**. Loading
   verifies and copies 2.35 GB of weights, which takes a few seconds.

You can also set the folder for a run without the picker: in *Product › Scheme ›
Edit Scheme › Run › Arguments*, add `-modelFolder /path/to/d1-omni-600M`, or set the
`D1_MODEL_FOLDER` environment variable.

## Run in the iOS Simulator

1. Start the router on the Mac from the repository root. Build with `xcodebuild`
   (see the main README), then:

   ```bash
   .build/xcode/Build/Products/Debug/system-one serve --port 8083 --model-folder <d1-omni-600M>
   ```

2. Choose an iPhone simulator destination and Run. The app connects to
   `http://127.0.0.1:8083/v1/systemone`, which the simulator reaches on the Mac.
   Change the address in Settings if you use another port.

The app asks the server one test question at launch, so a stopped server shows a
**Retry** banner instead of failing on the first message.

## Run on an iPhone or iPad

Set your team and a unique bundle identifier under *Signing & Capabilities*. The
app requests `com.apple.developer.kernel.increased-memory-limit`, which needs a
paid team. Copy the snapshot into the app's `Documents/d1-omni-600M` folder with
Finder's **Files** tab, or pick it with **Settings › Choose Folder…**.

## Launch arguments

Settings are stored in `UserDefaults`, so launch arguments override them for one run:

| Argument | Effect |
|---|---|
| `-router onDevice` / `-router server` | Where routing runs |
| `-serverURL <url>` | The `system-one serve` endpoint |
| `-modelFolder <path>` | The d1 snapshot folder |
| `-precision float32` / `-precision float16` | d1 precision |

Notes are kept in memory and cleared on restart.
