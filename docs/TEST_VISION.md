# Testing vision on DeepSeek-V4-Flash-Vision (EXL3 MixedK) from another machine

This walks through reaching a DSV4-Flash server running on a DGX Spark from
your workstation, sending images to it, and reading the numbers that tell you
the vision path is healthy. Everything here uses the OpenAI-compatible API
vLLM exposes, so any harness that speaks `/v1/chat/completions` with
`image_url` content parts works unchanged.

Verified 2026-09-02 on one GB10 with the vision + DSpark route
(`vllm` nightly, `SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}'`,
`GPU_MEM_UTIL=0.88`, `MAX_MODEL_LEN=65536`, tool calling and reasoning parsers on).

## 1. Start the server on the Spark

```bash
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  GPU_MEM_UTIL=0.88 MAX_MODEL_LEN=65536 \
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}' \
  bash scripts/serve_one_spark_dsv4.sh
```

The script adds `--enable-auto-tool-choice --tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4` by default. Agent harnesses need the first two
(they send `tool_choice: "auto"`; without the flags every request is a 400),
and the third moves the model's thinking out of `content` into
`reasoning_content`. `TOOL_CALL_PARSER=""` or `REASONING_PARSER=""` turns
either off. A conservative first boot on a new box can use `MAX_MODEL_LEN=16384`.

The server listens on `0.0.0.0:8899` and is ready when the log prints
`Application startup complete`. Load takes 11-13 minutes on a GB10 (the pack is
read once from disk; the serve script drops the page cache first). Keep one
copy of the model per Spark: two model processes on a 128 GB unified-memory box
will take each other down.

## 2. Open a port forward from your workstation

The Spark's port 8899 is not exposed publicly. Forward it over the same SSH
path you use to reach the box. Pick one:

**Brev CLI** (the instance name is what `brev ls` prints):

```bash
brev port-forward <instance-name> --port 8899:8899
```

**Plain SSH** (works with any host alias in your `~/.ssh/config`, including
the one Brev writes):

```bash
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8899:127.0.0.1:8899 <spark-host-alias>
```

Both commands hold the terminal; leave them running (or `nohup ... &`). On
Windows, run the command inside WSL. WSL forwards `localhost` to Windows, so
`http://localhost:8899` then works from PowerShell, a browser, or any Windows
harness. If your SSH config uses `ControlMaster`, add
`-o ControlMaster=no -o ControlPath=none` so the forward gets its own
connection; a forward requested through an existing multiplexed master can
exit silently.

Check the tunnel:

```bash
curl -s http://localhost:8899/v1/models
```

You should see `"id":"DSV4-Flash"` and the server's `max_model_len`. An empty
reply means the tunnel is not up yet (SSH takes 5-10 s to connect) or the
server is still loading.

## 3. Send an image

### The one-file probe

`scripts/vision_probe.py` needs only Python 3 and Pillow. With no arguments it
draws a test card (a red circle, a blue square, the word SPARK) and asks the
model to describe it, so you can check the vision path without hunting for an
image:

```bash
python scripts/vision_probe.py
python scripts/vision_probe.py photo.jpg "What is in this picture?"
python scripts/vision_probe.py photo.jpg "Read all text in the image." http://localhost:8899
```

It prints the answer and a receipt line:

```
[prompt_tokens=244 completion_tokens=200 wall=10.9s ~18.4 tok/s incl. prefill]
```

`prompt_tokens` includes the image: the 512x384 test card costs about 230
tokens, a 1024x1024 photo costs more. A correct answer names the red circle on
the left, the blue square on the right and the word SPARK.

### Raw curl

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<JSON | python -m json.tool
{
  "model": "DSV4-Flash",
  "max_tokens": 300,
  "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG"}},
    {"type": "text", "text": "Describe this image."}
  ]}]
}
JSON
```

### From an OpenAI-compatible harness

Point the client at the tunnel; no key is checked:

| Setting | Value |
|---|---|
| base URL | `http://localhost:8899/v1` |
| model | `DSV4-Flash` |
| API key | any non-empty string |
| images | `image_url` content parts, `data:` URLs or public `http(s)` URLs |

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="x")
r = client.chat.completions.create(model="DSV4-Flash", max_tokens=300, messages=[
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "What does the sign say?"}]}])
print(r.choices[0].message.content, r.usage)
```

Several images in one message are fine; each one adds its own prompt tokens.

### Hermes Agent and Open WebUI

Both work against the tunnel with the settings above. Two things to know:

- **Hermes Agent** (`~/.hermes/config.yaml`, or the desktop app's
  `config.yaml`): `model.provider: custom`, `base_url: http://localhost:8899/v1`,
  `api_key: x`, `context_length: 65536` (Hermes refuses models under 64k) and
  `max_tokens: 8192`. Without `max_tokens` Hermes asks for the whole context
  as output and the server answers 400 "requested 65536 output tokens".
  Hermes analyzes an image through its `vision_analyze` tool, a separate
  model call carrying the full image: a 2 MB phone photo plus the model's
  thinking takes 60-100 s on one Spark. Resize to ~1024 px on the long side,
  or ask for a one-line answer, for a much faster turn.
- **Open WebUI**: `OPENAI_API_BASE_URL=http://127.0.0.1:8899/v1`,
  `OPENAI_API_KEY=x`; the model appears as `DSV4-Flash` and images attach
  through the normal chat box.

## 4. What a healthy server looks like

| Check | Expected on one GB10 (vision + DSpark, util 0.88, 16k ctx) |
|---|---|
| `/v1/models` | `DSV4-Flash`, `max_model_len` as served |
| Test card | red circle, blue square, "SPARK" all named |
| Prompt tokens, 512x384 image + short question | ~240 |
| Decode with an image in context | 13.5-19.8 tok/s |
| Decode, text only, 256 tokens | ~19.7 tok/s (draft acceptance 2.3-3.2 on short answers, ~1.9 on long ones) |
| Server log, KV cache | `GPU KV cache size: 298,380-328,319 tokens` at util 0.88, 64k (84,554 at 16k) |
| Tool call round trip | `finish_reason: tool_calls`, parsed arguments, ~12 s including the thinking |
| Spark `MemAvailable` while serving | ~9 GiB |

The model reasons before it answers. Short `max_tokens` (under ~150) can cut
the reply off inside that reasoning, which looks like the model "listing
drafts" instead of answering. Give it 300-400 tokens for a description, or
ask for a one-line answer.

## 5. Context length

`MAX_MODEL_LEN` is a per-request cap, not a memory setting. The KV pool is
sized from `GPU_MEM_UTIL` after the model loads, and the pool decides how much
context you can actually hold:

| Serve shape | KV pool | Fits |
|---|---|---|
| vision + DSpark, util 0.88, `MAX_MODEL_LEN=16384` | 84,554 tokens | five 16k requests |
| vision + DSpark, util 0.88, `MAX_MODEL_LEN=65536` | 298,380-328,319 tokens | four or five 64k requests |
| vision, no draft, util 0.85, 16k | 151,575 tokens | nine 16k requests |

On this nightly the pool grows with `max-model-len` (the block allocation is
sized against the cap), so raising the cap costs no host memory: the 64k boot
held `MemAvailable` at 8.3-9.1 GiB, the same as 16k, and a 47,947-token
prompt prefilled in 148 s (~324 tok/s) with no watchdog action. 131072 is
the next step and has not been measured yet.
Anything beyond the pool size fails at startup with a clear
"max seq len is larger than the KV cache" error, not at request time. Longer
prompts do use more prefill scratch (attention indexer logits and the sliced
window-attention partials), so when you raise the cap, keep the memory
watchdog running and confirm `MemAvailable` on the first long prompt.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `curl` returns nothing, tunnel command exited | Port 8899 already forwarded (a second forward exits with `ExitOnForwardFailure`), or the forward went through a multiplexed SSH master; relaunch with `ControlMaster=no` |
| HTTP 400 `maximum context length` | Image plus prompt exceeds `MAX_MODEL_LEN`; downscale the image or raise the cap (section 5) |
| Answer is a list of "Draft:" lines | `max_tokens` too small for the reasoning plus the answer; raise it |
| `ssh` banner timeout after starting the server | The box is memory-wedged; check `MemAvailable` before starting anything else |
| Text works, images 500 | The wide-SWA prefill patch is missing (`scripts/patch_dsv4_vl_sm120_wide_swa.py`); the log shows `Unsupported sparse-MLA prefill configuration ... topk=512` |
