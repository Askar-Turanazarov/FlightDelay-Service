// Opens the live demo like a visitor so Streamlit Cloud does not put it to sleep.
import { chromium } from "playwright";

const URL = "https://flight-delay-askar.streamlit.app";
const TIMEOUT_MS = 5 * 60 * 1000;
const READY_TEXT = /flight|рейс|parvoz/i;
const WAKE_BUTTON = /get this app back up/i;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const started = Date.now();

try {
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

  while (Date.now() - started < TIMEOUT_MS) {
    const wake = page.getByRole("button", { name: WAKE_BUTTON });
    if (await wake.isVisible().catch(() => false)) {
      console.log("App was asleep, pressing the wake button");
      await wake.click();
    }

    // The app itself renders inside an iframe; the shell page alone proves nothing.
    for (const frame of page.frames()) {
      if (frame === page.mainFrame()) continue;
      const text = await frame.locator("[data-testid='stApp']").innerText({ timeout: 2_000 }).catch(() => "");
      if (READY_TEXT.test(text)) {
        console.log(`Awake after ${Math.round((Date.now() - started) / 1000)} s`);
        process.exit(0);
      }
    }
    await page.waitForTimeout(5_000);
  }
  throw new Error(`App text did not appear within ${TIMEOUT_MS / 1000} s`);
} catch (error) {
  await page.screenshot({ path: "keep-awake-failure.png", fullPage: true }).catch(() => {});
  console.error(error.message);
  process.exit(1);
} finally {
  await browser.close();
}
