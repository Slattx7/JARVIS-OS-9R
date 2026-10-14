# Unified Jarvis 9R Agent Rules

## Critical: Screen Captures over Network ADB
When capturing screenshots of the device (`100.88.162.69:5555` or any network ADB device):
- **NEVER** execute `adb exec-out screencap -p > <file>.png`.
  Streaming binary PNG data across network sockets (Tailscale) causes Android's encoder to abort with `Failed to compress PNG (error code: -2)`. Shell redirection dumps this text into the image file, producing a truncated PNG that fatally crashes the conversation with an unrecoverable `HTTP 400 Bad Request (INVALID_ARGUMENT)`.
- **ALWAYS** capture to local device storage first, then pull:
  ```bash
  adb -s 100.88.162.69:5555 shell screencap -p /sdcard/screencap.png && adb -s 100.88.162.69:5555 pull /sdcard/screencap.png <destination.png>
  ```
- Verify the file is valid before using `view_file`:
  ```bash
  python3 -c "from PIL import Image; Image.open('<destination.png>').verify()"
  ```
