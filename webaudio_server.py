"""Local-only Web Audio comparison server; writes only fixed files under outputs/."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent
OUT = ROOT / 'outputs'
FS = 48000
# Python reference: `python render.py --output outputs/hrtf_left_to_right_smooth.wav`
# (default seeded noise and trajectory match webaudio_compare.html).
REFERENCE = OUT/'hrtf_left_to_right_smooth.wav'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != '/save-webaudio':
            self.send_error(404)
            return
        size = int(self.headers.get('Content-Length', 0))
        if size != round(5.1*FS)*2*4:
            self.send_error(400, 'Unexpected audio size')
            return
        audio = np.frombuffer(self.rfile.read(size), dtype='<f4').reshape(-1, 2).copy()
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
            self.send_error(400, 'Invalid or silent render')
            return
        sf.write(OUT/'webaudio_left_to_right_raw.wav', audio, FS, subtype='FLOAT')
        # Same broadband RMS over the active five seconds, one gain per stereo file.
        report = {}
        for name, y in [('webaudio_left_to_right_matched', audio),
                        ('python_left_to_right_matched', sf.read(REFERENCE)[0])]:
            rms = np.sqrt(np.mean(y[:5*FS]**2))
            gain = 10**(-24/20)/rms
            if np.max(np.abs(y))*gain >= .99:
                self.send_error(500, 'Matching target clips')
                return
            sf.write(OUT/(name+'.wav'), y*gain, FS, subtype='PCM_24')
            report[name] = {'gain': float(gain), 'rms_dbfs': -24, 'peak': float(np.max(np.abs(y))*gain)}
        (OUT/'webaudio_comparison.json').write_text(json.dumps(report, indent=2))
        payload = json.dumps(report).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == '__main__':
    if not REFERENCE.is_file():
        raise SystemExit(f'Missing {REFERENCE}; run: python render.py --output {REFERENCE.relative_to(ROOT).as_posix()}')
    OUT.mkdir(exist_ok=True)
    x = np.random.default_rng(42).normal(0, .08, 5*FS)
    fade = round(.03*FS)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    (OUT/'comparison_input.f32').write_bytes(x.astype('<f4').tobytes())
    print('http://127.0.0.1:8766/webaudio_compare.html', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
