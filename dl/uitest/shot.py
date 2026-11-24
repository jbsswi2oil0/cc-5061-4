import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

SHOTS = "/workspace/reports"
os.makedirs(SHOTS, exist_ok=True)

result = {"console_errors": [], "page_errors": [], "checks": {}}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: result["console_errors"].append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: result["page_errors"].append(str(e)))

        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        page.wait_for_selector(".sample-chip", timeout=15000)
        page.wait_for_function(
            "() => document.querySelectorAll('.sample-chip').length === 3",
            timeout=15000)
        result["checks"]["sample_chips"] = page.locator(".sample-chip").count()

        # auto-loaded planar job: wait for end-to-end metrics
        page.wait_for_function(
            "() => { const t=document.getElementById('mHfOut').textContent;"
            " return t && t !== '–'; }",
            timeout=240000,
        )
        page.wait_for_timeout(2500)
        result["checks"]["hf_orig"] = page.inner_text("#mHfOrig")
        result["checks"]["hf_out"] = page.inner_text("#mHfOut")
        result["checks"]["hf_sub"] = page.inner_text("#mHfSub").strip()
        result["checks"]["fov"] = page.inner_text("#mFov")
        result["checks"]["sac"] = page.inner_text("#mSac")
        result["checks"]["rmse"] = page.inner_text("#mRmse")
        result["checks"]["stage"] = page.inner_text("#stageText")

        # video element decode status (H.264 availability in this browser)
        result["checks"]["src_video"] = page.evaluate(
            """() => { const v=document.getElementById('srcVideo');
               return {readyState:v.readyState, vw:v.videoWidth, vh:v.videoHeight,
                       src: !!v.src, duration: v.duration}; }"""
        )
        result["checks"]["stab_video"] = page.evaluate(
            """() => { const v=document.getElementById('stabVideo');
               return {readyState:v.readyState, vw:v.videoWidth, vh:v.videoHeight,
                       src: !!v.src, duration: v.duration}; }"""
        )

        # canvas non-blank pixel probe
        result["checks"]["chart_pixels"] = page.evaluate(
            """() => {
              function nonblank(id){
                const c=document.getElementById(id);
                const x=c.getContext('2d');
                const d=x.getImageData(0,0,c.width,c.height).data;
                let n=0; for(let i=3;i<d.length;i+=4){ if(d[i]>0) n++; }
                return {w:c.width,h:c.height,painted:n};
              }
              return {traj:nonblank('chartTraj'), fov:nonblank('chartFov'),
                      resid:nonblank('chartResid'), overlay:nonblank('overlay')};
            }"""
        )

        page.screenshot(path=os.path.join(SHOTS, "ui_planar_top.png"))
        page.screenshot(path=os.path.join(SHOTS, "ui_planar_full.png"),
                        full_page=True)

        # playback / sync check
        try:
            page.click("#playBtn", timeout=3000)
            page.wait_for_timeout(1800)
            result["checks"]["playback"] = page.evaluate(
                """() => { const a=document.getElementById('srcVideo');
                   const b=document.getElementById('stabVideo');
                   return {srcT:a.currentTime, stabT:b.currentTime,
                           srcPaused:a.paused, stabPaused:b.paused}; }"""
            )
            page.screenshot(path=os.path.join(SHOTS, "ui_playing.png"))
            page.click("#playBtn")
        except Exception as e:
            result["checks"]["playback_error"] = str(e)

        # seek to middle -> overlay polygon should move
        page.eval_on_selector("#seekBar", "el => { el.value=500;"
                              "el.dispatchEvent(new Event('input',{bubbles:true})); }")
        page.wait_for_timeout(600)
        result["checks"]["seek_time"] = page.inner_text("#timeLabel")
        page.screenshot(path=os.path.join(SHOTS, "ui_seek_mid.png"))

        # switch trajectory channel
        page.click('#trajSeg button[data-ch="a"]')
        page.wait_for_timeout(300)

        # parallax sample -> warning banner
        page.locator(".sample-chip", has_text="视差").first.click()
        page.wait_for_function(
            "() => { const b=document.getElementById('parallaxBanner');"
            " return b && !b.classList.contains('hidden')"
            " && b.innerText.includes('视差'); }",
            timeout=240000,
        )
        page.wait_for_timeout(1500)
        result["checks"]["parallax_banner"] = page.inner_text("#parallaxBanner").strip()[:220]
        result["checks"]["parallax_class"] = page.get_attribute(
            "#parallaxBanner", "class")

        # push smoothness + FOV sliders -> kernel plan preview
        page.eval_on_selector("#smoothRange",
                              "el => { el.value=96; el.dispatchEvent(new Event('input',{bubbles:true})); }")
        page.eval_on_selector("#fovRange",
                              "el => { el.value=95; el.dispatchEvent(new Event('input',{bubbles:true})); }")
        page.wait_for_function(
            "() => document.getElementById('mHfSub').innerText.includes('规划目标')",
            timeout=10000,
        )
        page.wait_for_timeout(800)
        result["checks"]["tightfov_fov"] = page.inner_text("#mFov")
        result["checks"]["tightfov_sac"] = page.inner_text("#mSac")
        page.screenshot(path=os.path.join(SHOTS, "ui_parallax_tightfov.png"),
                        full_page=True)

        # benchmark table populated from reports/benchmark.json
        page.wait_for_selector("#benchTable tbody tr", timeout=5000)
        result["checks"]["bench_rows"] = page.locator(
            "#benchTable tbody tr").count()

        browser.close()

    # de-duplicate noisy media errors
    result["console_errors"] = [e for e in result["console_errors"]
                               if "favicon" not in e][:15]
    out = "/workspace/reports/browser_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    if result["page_errors"]:
        print("PAGE ERRORS:", result["page_errors"][:5])
        sys.exit(2)
    print("BROWSER_CHECK_OK")


if __name__ == "__main__":
    main()
