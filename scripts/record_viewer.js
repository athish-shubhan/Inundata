// Renders the viewer's guided tour to docs/viewer.mp4 and a highlight to docs/viewer.gif.
//
//   (cd docs/viewer && python3 -m http.server 8765) &
//   node scripts/record_viewer.js
//
// Needs Node with Playwright and ffmpeg on PATH. The page is stepped one frame
// at a time with a supplied clock (?capture), so the result is smooth even on a
// machine without a GPU.
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const URL = process.env.VIEWER_URL || 'http://127.0.0.1:8765/?capture';
const FPS = 24, WIDTH = 1280, HEIGHT = 720;
const docs = path.join(__dirname, '..', 'docs');

(async () => {
  const frames = fs.mkdtempSync(path.join(os.tmpdir(), 'inundata-frames-'));
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  await page.goto(URL);
  await page.waitForFunction(() => typeof window.inundataFrame === 'function', null, { timeout: 60000 });

  let n = 0;
  for (;; n++) {
    const running = await page.evaluate((ms) => window.inundataFrame(ms), 1000 + (n * 1000) / FPS);
    if (!running) break;
    await page.screenshot({ path: path.join(frames, `f${String(n).padStart(5, '0')}.jpg`), quality: 93 });
    if (n % 100 === 0) console.log(`frame ${n}`);
  }
  await browser.close();
  console.log(`${n} frames (${(n / FPS).toFixed(1)} s)`);

  const input = ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(frames, 'f%05d.jpg')];
  execFileSync('ffmpeg', [...input, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart',
    path.join(docs, 'viewer.mp4')]);
  // The README GIF is a 12 s highlight (observed flood, then the terrain baseline);
  // a GIF of the whole tour is over 12 MB. The mp4 holds the full tour.
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', '4.6', '-t', '12.2', '-i', path.join(docs, 'viewer.mp4'), '-vf',
    'fps=8,scale=680:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=72:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
    path.join(docs, 'viewer.gif')]);
  fs.rmSync(frames, { recursive: true, force: true });
})();
