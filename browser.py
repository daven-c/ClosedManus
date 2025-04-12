import logging
from typing import Dict, Any, List, Optional  # Added Optional
from playwright.async_api import async_playwright, Page
import base64  # Import base64

logger = logging.getLogger("web_automation")


class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def initialize(self) -> bool:
        """Initialize the browser with robust error handling"""
        try:
            # Make sure we start fresh
            await self.close()

            logger.info("Starting Playwright...")
            self.playwright = await async_playwright().start()

            browser_type = "chromium"
            logger.info(f"Launching {browser_type}...")
            browser_launcher = getattr(self.playwright, browser_type)

            # Launch with appropriate options
            self.browser = await browser_launcher.launch(headless=False)
            self.page = await self.browser.new_page()
            await self.page.goto("about:blank", timeout=5000)
            logger.info(f"Browser initialized using {browser_type}")
            return True
        except Exception as e:
            logger.error(f"Browser initialization error: {e}")
            return False

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return {"success": True, "message": f"Navigated to {url}"}
        except Exception as e:
            return {"success": False, "message": f"Navigation error: {e}"}

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click on an element"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.click(selector)
            return {"success": True, "message": f"Clicked on {selector}"}
        except Exception as e:
            return {"success": False, "message": f"Click error: {e}"}

    async def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text into an element"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.fill(selector, text)
            return {"success": True, "message": f"Typed text into {selector}"}
        except Exception as e:
            return {"success": False, "message": f"Type error: {e}"}

    async def get_page_content(self) -> Dict[str, Any]:
        """Get the current page content"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            url = self.page.url
            title = await self.page.title()
            content = await self.page.content()
            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content[:5000]  # Limit content size
            }
        except Exception as e:
            return {"success": False, "message": f"Error getting content: {e}"}

    async def get_page_details(self) -> Dict[str, Any]:
        """Get current page details including URL, title, and full HTML content."""
        if not self.page or self.page.is_closed():
            return {"success": False, "message": "Browser not initialized or page closed"}

        try:
            url = self.page.url
            title = await self.page.title()

            # Get full HTML content
            html_content = await self.page.content()
            # Limit HTML size slightly to avoid excessively large prompts, but keep most context
            MAX_HTML_LENGTH = 20000
            if len(html_content) > MAX_HTML_LENGTH:
                logger.warning(
                    f"HTML content length ({len(html_content)}) exceeds limit ({MAX_HTML_LENGTH}), truncating.")
                html_content = html_content[:MAX_HTML_LENGTH] + \
                    "\n... (truncated)"

            # Keep interactable extraction for potential future use or different LLM strategies,
            # but we won't pass it directly in the main action prompt anymore.
            interactables = []
            try:
                # Find buttons
                buttons = await self.page.locator('button:visible, input[type="button"]:visible, input[type="submit"]:visible, [role="button"]:visible').all()
                for i, btn in enumerate(buttons):
                    text = await btn.text_content() or await btn.get_attribute('aria-label') or await btn.get_attribute('value')
                    selector = f'button:visible, input[type="button"]:visible, input[type="submit"]:visible, [role="button"]:visible >> nth={i}'
                    interactables.append({
                        "type": "button",
                        "text": text.strip() if text else "[no text]",
                        "selector": selector
                    })
                # ... (similar extraction for links and inputs - can be kept or removed if not used)
            except Exception as ie:
                logger.warning(
                    f"Could not extract all interactable elements: {ie}")

            return {
                "success": True,
                "url": url,
                "title": title,
                "html_content": html_content,  # Use full HTML content
                # "interactable_elements": interactables # Keep data but don't use in main prompt
            }
        except Exception as e:
            logger.error(f"Error getting page details: {e}", exc_info=True)
            return {"success": False, "message": f"Error getting page details: {e}"}

    async def get_screenshot(self, full_page: bool = False) -> Optional[str]:
        """Capture a screenshot of the current page and return as base64."""
        if not self.page or self.page.is_closed():
            logger.warning("Cannot get screenshot, page not available.")
            return None
        try:
            screenshot_bytes = await self.page.screenshot(full_page=full_page, type='png')
            base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None

    async def close(self):
        """Close browser resources"""
        try:
            if self.page and not self.page.is_closed():
                await self.page.close()
        except:
            pass

        try:
            if self.browser:
                await self.browser.close()
        except:
            pass

        try:
            if self.playwright:
                await self.playwright.stop()
        except:
            pass

        self.page = None
        self.browser = None
        self.playwright = None
        logger.info("Browser resources closed")
