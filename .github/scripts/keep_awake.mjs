// Opens the live demo like a visitor so Streamlit Cloud does not put it to sleep, once per demo model:
// the first visit wakes the app (the default weather model), the second loads the diploma model into the
// shared cache, so a real visitor who switches models does not wait for the load.
import { chromium } from "playwright";

const BASE = "https://flight-delay-askar.streamlit.app";
const MODELS = ["weather", "compact"];
const TIMEOUT_MS = 5 * 60 * 1000;
const READY_TEXT = /flight|рейс|parvoz/i;
const WAKE_BUTTON = /get this app back up/i;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

async function visit(model) {
  const started = Date.now();
  await page.goto(`${BASE}/?model=${model}`, { waitUntil: "domcontentloaded", timeout: 60_000 });

  while (Date.now() - started < TIMEOUT_MS) {
    const wake = page.getByRole("button", { name: WAKE_BUTTON });
    if (await wake.isVisible().catch(() => false)) {
      console.log("App was asleep, pressing the wake button");
      await wake.click();
    }

    // The app itself renders inside an iframe; the shell page alone proves nothing. The model loads after the
    // title, so the app is ready for this model once the analysis button is on the page.
    for (const frame of page.frames()) {
      if (frame === page.mainFrame()) continue;
      const app = frame.locator("[data-testid='stApp']");
      const text = await app.innerText({ timeout: 2_000 }).catch(() => "");
      const ready = await app.locator("[data-testid='stButton']").count().catch(() => 0);
      if (READY_TEXT.test(text) && ready > 0) {
        console.log(`${model}: ready after ${Math.round((Date.now() - started) / 1000)} s`);
        return;
      }
    }
    await page.waitForTimeout(5_000);
  }
  throw new Error(`${model}: app text did not appear within ${TIMEOUT_MS / 1000} s`);
}

try {
  for (const model of MODELS) await visit(model);
} catch (error) {
  await page.screenshot({ path: "keep-awake-failure.png", fullPage: true }).catch(() => {});
  console.error(error.message);
  process.exitCode = 1;
} finally {
  await browser.close();
}
