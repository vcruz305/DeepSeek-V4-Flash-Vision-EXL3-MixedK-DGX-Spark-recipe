"""vision_probe.py: send one image to the DSV4-Flash server and print the answer + speed.
Usage: python vision_probe.py [image_path] [prompt] [base_url]
Without an image path it draws a test card (red circle, blue square, the word SPARK)."""
import base64, io, json, sys, time, urllib.request
img, prompt = (sys.argv[1] if len(sys.argv) > 1 else None), (sys.argv[2] if len(sys.argv) > 2 else "Describe this image in two sentences. Name every shape, colour and any text.")
base = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8899"
if img:
    data = open(img, "rb").read(); mime = "image/png" if img.lower().endswith(".png") else "image/jpeg"
else:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 384), "white"); d = ImageDraw.Draw(im)
    d.ellipse((40, 60, 200, 220), fill="red"); d.rectangle((300, 80, 460, 240), fill="blue"); d.text((180, 300), "SPARK", fill="black")
    buf = io.BytesIO(); im.save(buf, "PNG"); data, mime = buf.getvalue(), "image/png"
b64 = base64.b64encode(data).decode()
body = {"model": "DSV4-Flash", "max_tokens": 400, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt}]}]}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
u = r["usage"]; print(r["choices"][0]["message"]["content"].strip())
print(f"\n[prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} wall={dt:.1f}s ~{u['completion_tokens']/dt:.1f} tok/s incl. prefill]")
