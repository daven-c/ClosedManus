import logging
import asyncio  # Import asyncio
from typing import Dict, Any, List  # Added List
from playwright.async_api import async_playwright, Page  # Added Page

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
            # Add https:// protocol if missing
            if not url.startswith(('http://', 'https://')):
                logger.info(f"Adding https:// prefix to URL: {url}")
                url = f"https://{url}"

            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return {"success": True, "message": f"Navigated to {url}"}
        except Exception as e:
            logger.error(f"Navigation error: {e}", exc_info=True)
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

    async def execute_javascript(self, code: str) -> Dict[str, Any]:
        """Execute JavaScript code in the browser context"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            result = await self.page.evaluate(code)
            return {
                "success": True,
                "message": "JavaScript executed successfully",
                "result": result
            }
        except Exception as e:
            return {"success": False, "message": f"JavaScript execution error: {e}"}

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
            # Add a short wait for network idle to potentially catch more dynamic content
            try:
                # Wait max 1.5s
                await self.page.wait_for_load_state('networkidle', timeout=1500)
                logger.debug("Waited for networkidle state.")
            except Exception as wait_error:
                logger.debug(
                    f"Networkidle wait timed out or failed: {wait_error}. Proceeding anyway.")
                await asyncio.sleep(0.5)  # Fallback short sleep

            url = self.page.url
            title = await self.page.title()

            # Get full HTML content
            html_content = await self.page.content()
            original_length = len(html_content)

            # Limit HTML size slightly to avoid excessively large prompts, but keep most context
            MAX_HTML_LENGTH = 20000  # Keep this limit relatively high here
            truncated_length = original_length
            if original_length > MAX_HTML_LENGTH:
                logger.warning(
                    f"HTML content length ({original_length}) exceeds browser limit ({MAX_HTML_LENGTH}), truncating.")
                html_content = html_content[:MAX_HTML_LENGTH] + \
                    "\n... (truncated in browser.py)"
                truncated_length = len(html_content)

            # Log original and truncated lengths
            logger.debug(
                f"HTML Length: Original={original_length}, Returned={truncated_length}")

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

    async def scan_actionable_elements(self) -> Dict[str, Any]:
        """Scans the page using JavaScript to identify all actionable elements and their properties"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            scan_script = """
            function getSelectorPath(element) {
                const path = [];
                while (element && element.nodeType === Node.ELEMENT_NODE) {
                    let selector = element.tagName.toLowerCase();
                    
                    if (element.id) {
                        selector += '#' + CSS.escape(element.id);
                        path.unshift(selector);
                        break;
                    }
                    
                    let sibling = element;
                    let nth = 1;
                    while (sibling = sibling.previousElementSibling) {
                        if (sibling.tagName === element.tagName) nth++;
                    }
                    
                    if (nth > 1) {
                        selector += `:nth-of-type(${nth})`;
                    }
                    
                    path.unshift(selector);
                    element = element.parentElement;
                }
                
                return path.join(' > ');
            }

            function getAdditionalAttributes(element) {
                const attributes = {};
                const importantAttrs = ['name', 'class', 'type', 'role', 'aria-label', 'placeholder', 'value', 'href', 'data-test-id'];
                
                for (const attr of importantAttrs) {
                    if (element.hasAttribute(attr)) {
                        attributes[attr] = element.getAttribute(attr);
                    }
                }
                
                return attributes;
            }

            function isElementVisible(element) {
                if (!element.getBoundingClientRect) return false;
                
                const rect = element.getBoundingClientRect();
                const style = window.getComputedStyle(element);
                
                return (
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    style.opacity !== '0' &&
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.top < window.innerHeight &&
                    rect.left < window.innerWidth &&
                    rect.bottom > 0 &&
                    rect.right > 0
                );
            }

            function scanActionableElements() {
                const actionableElements = {
                    buttons: [],
                    links: [],
                    inputs: [],
                    selects: [],
                    checkboxes: [],
                    radioButtons: [],
                    dropdowns: [],
                    other: []
                };
                
                // Get all potentially interactive elements
                const interactives = document.querySelectorAll(
                    'button, a, input, select, textarea, [role="button"], [role="link"], ' +
                    '[role="checkbox"], [role="radio"], [role="menuitem"], [role="tab"], ' +
                    '[onclick], [tabindex="0"], .btn, .button, [aria-haspopup]'
                );
                
                interactives.forEach(element => {
                    if (!isElementVisible(element)) return;
                    
                    const tagName = element.tagName.toLowerCase();
                    const role = element.getAttribute('role');
                    const textContent = element.textContent ? element.textContent.trim() : '';
                    const selector = getSelectorPath(element);
                    const rect = element.getBoundingClientRect();
                    const attributes = getAdditionalAttributes(element);
                    
                    const elementInfo = {
                        type: tagName,
                        text: textContent,
                        selector: selector,
                        role: role || null,
                        attributes: attributes,
                        position: {
                            x: rect.left,
                            y: rect.top,
                            width: rect.width,
                            height: rect.height
                        }
                    };
                    
                    // Categorize element
                    if (tagName === 'button' || role === 'button' || element.classList.contains('btn') || element.classList.contains('button')) {
                        actionableElements.buttons.push(elementInfo);
                    } else if (tagName === 'a' || role === 'link') {
                        actionableElements.links.push(elementInfo);
                    } else if (tagName === 'input') {
                        const inputType = element.getAttribute('type') || 'text';
                        if (inputType === 'checkbox') {
                            actionableElements.checkboxes.push({...elementInfo, checked: element.checked});
                        } else if (inputType === 'radio') {
                            actionableElements.radioButtons.push({...elementInfo, checked: element.checked});
                        } else if (['text', 'email', 'number', 'password', 'search', 'tel', 'url'].includes(inputType)) {
                            actionableElements.inputs.push({...elementInfo, value: element.value});
                        } else {
                            actionableElements.other.push(elementInfo);
                        }
                    } else if (tagName === 'select' || role === 'listbox') {
                        actionableElements.selects.push(elementInfo);
                    } else if (role === 'menuitem' || element.getAttribute('aria-haspopup')) {
                        actionableElements.dropdowns.push(elementInfo);
                    } else {
                        actionableElements.other.push(elementInfo);
                    }
                });
                
                // Add page metadata
                const metadata = {
                    url: window.location.href,
                    title: document.title,
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    }
                };
                
                return {
                    metadata: metadata,
                    actionableElements: actionableElements,
                    stats: {
                        buttons: actionableElements.buttons.length,
                        links: actionableElements.links.length,
                        inputs: actionableElements.inputs.length,
                        selects: actionableElements.selects.length,
                        checkboxes: actionableElements.checkboxes.length,
                        radioButtons: actionableElements.radioButtons.length,
                        dropdowns: actionableElements.dropdowns.length,
                        other: actionableElements.other.length,
                        total: Object.values(actionableElements).reduce((sum, arr) => sum + arr.length, 0)
                    }
                };
            }
            
            return scanActionableElements();
            """

            result = await self.page.evaluate(scan_script)
            return {
                "success": True,
                "message": "Successfully scanned page for actionable elements",
                "scan_result": result
            }
        except Exception as e:
            logger.error(
                f"Error scanning actionable elements: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"JavaScript scan error: {e}"
            }

    async def detect_error_state(self) -> Dict[str, Any]:
        """Detects if the page is in an error state and identifies the type of error"""
        if not self.page:
            return {"error_detected": False, "message": "Browser not initialized"}

        try:
            # Execute JavaScript to detect common error patterns
            detection_script = """
            function detectErrors() {
                const results = {
                    hasError: false,
                    errorType: null,
                    errorMessage: null,
                    possibleRecoveryActions: []
                };
                
                // Check for HTTP error codes in text
                const errorCodes = ['400', '401', '403', '404', '500', '502', '503', '504'];
                const pageText = document.body.innerText;
                
                for (const code of errorCodes) {
                    if (
                        pageText.includes(`Error ${code}`) || 
                        pageText.includes(`${code} Error`) || 
                        pageText.includes(`HTTP ${code}`)
                    ) {
                        results.hasError = true;
                        results.errorType = 'http_error';
                        results.errorMessage = `HTTP Error ${code} detected on page`;
                        results.possibleRecoveryActions.push('reload_page');
                        if (code === '401' || code === '403') {
                            results.possibleRecoveryActions.push('check_authentication');
                        }
                    }
                }
                
                // Check for connection errors
                const connectionErrorPatterns = [
                    'connection failed', 
                    'cannot connect', 
                    'network error',
                    'failed to connect',
                    'no internet',
                    'offline',
                    'could not reach',
                    'ERR_CONNECTION'
                ];
                
                for (const pattern of connectionErrorPatterns) {
                    if (pageText.toLowerCase().includes(pattern.toLowerCase())) {
                        results.hasError = true;
                        results.errorType = 'connection_error';
                        results.errorMessage = 'Connection error detected on page';
                        results.possibleRecoveryActions.push('reload_page', 'wait_and_retry');
                    }
                }
                
                // Check for CAPTCHA or verification challenges
                const captchaPatterns = [
                    'captcha',
                    'robot check',
                    'human verification',
                    'prove you are human',
                    'security check',
                    'automated access',
                    'suspicious activity'
                ];
                
                for (const pattern of captchaPatterns) {
                    if (pageText.toLowerCase().includes(pattern.toLowerCase())) {
                        results.hasError = true;
                        results.errorType = 'captcha';
                        results.errorMessage = 'CAPTCHA or verification challenge detected';
                        results.possibleRecoveryActions.push('pause_for_user');
                    }
                }
                
                // Check for login walls or authentication challenges
                const loginPatterns = [
                    'please sign in',
                    'please log in',
                    'login required',
                    'sign in to continue',
                    'login to continue',
                    'authentication required'
                ];
                
                for (const pattern of loginPatterns) {
                    if (pageText.toLowerCase().includes(pattern.toLowerCase()) &&
                        (document.querySelector('input[type="password"]') || 
                         document.querySelector('input[type="email"]') ||
                         document.querySelector('input[name="username"]'))) {
                        results.hasError = true;
                        results.errorType = 'authentication_required';
                        results.errorMessage = 'Login or authentication required';
                        results.possibleRecoveryActions.push('authentication_flow');
                    }
                }
                
                // Check for popup/modal dialogs that might be blocking interaction
                const modalElements = document.querySelectorAll('[role="dialog"], .modal, .popup, .overlay');
                if (modalElements.length > 0) {
                    for (const modal of modalElements) {
                        if (window.getComputedStyle(modal).display !== 'none' && 
                            window.getComputedStyle(modal).visibility !== 'hidden') {
                            results.hasError = true;
                            results.errorType = 'modal_dialog';
                            results.errorMessage = 'Modal dialog or popup detected';
                            results.possibleRecoveryActions.push('dismiss_modal');
                            
                            // Check if it might be a cookie consent dialog
                            const modalText = modal.innerText.toLowerCase();
                            if (modalText.includes('cookie') || 
                                modalText.includes('privacy') || 
                                modalText.includes('consent') || 
                                modalText.includes('accept')) {
                                results.errorType = 'cookie_consent';
                                results.errorMessage = 'Cookie consent dialog detected';
                                results.possibleRecoveryActions.push('accept_cookies');
                            }
                        }
                    }
                }
                
                return results;
            }
            
            return detectErrors();
            """

            detection_result = await self.page.evaluate(detection_script)

            # Add page URL to the information
            detection_result["current_url"] = self.page.url

            return detection_result

        except Exception as e:
            logger.error(f"Error detecting page state: {e}", exc_info=True)
            return {"error_detected": False, "message": f"Error detection failed: {str(e)}"}

    async def attempt_recovery(self, error_state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempts to recover from detected error states automatically"""
        if not error_state.get("hasError", False):
            return {"success": True, "message": "No error to recover from"}

        error_type = error_state.get("errorType")
        recovery_actions = error_state.get("possibleRecoveryActions", [])

        logger.info(
            f"Attempting recovery from {error_type} error with actions: {recovery_actions}")

        try:
            if "reload_page" in recovery_actions:
                logger.info("Recovery action: Reloading page")
                await self.page.reload(wait_until="domcontentloaded", timeout=30000)
                return {"success": True, "message": "Page reloaded as recovery action", "action_taken": "reload_page"}

            elif "dismiss_modal" in recovery_actions:
                logger.info("Recovery action: Attempting to dismiss modal")
                # Try common dismiss methods
                dismiss_script = """
                function dismissModal() {
                    // Try to find close buttons in dialogs
                    const closeButtons = Array.from(document.querySelectorAll(
                        '.close, .close-button, .dismiss, [aria-label="Close"], [aria-label="Dismiss"], button.btn-close'
                    ));
                    
                    // Filter to only visible buttons
                    const visibleCloseButtons = closeButtons.filter(btn => {
                        const style = window.getComputedStyle(btn);
                        return style.display !== 'none' && 
                               style.visibility !== 'hidden' &&
                               btn.offsetWidth > 0 &&
                               btn.offsetHeight > 0;
                    });
                    
                    if (visibleCloseButtons.length > 0) {
                        visibleCloseButtons[0].click();
                        return { clicked: true, element: 'close_button' };
                    }
                    
                    // Try to find and click an overlay backdrop
                    const overlays = Array.from(document.querySelectorAll(
                        '.modal-backdrop, .overlay, .dialog-backdrop'
                    ));
                    
                    const visibleOverlays = overlays.filter(overlay => {
                        const style = window.getComputedStyle(overlay);
                        return style.display !== 'none' && 
                               style.visibility !== 'hidden';
                    });
                    
                    if (visibleOverlays.length > 0) {
                        visibleOverlays[0].click();
                        return { clicked: true, element: 'overlay' };
                    }
                    
                    // Try hitting Escape key as a last resort
                    // (can't directly do this in JS, will need to use Playwright's keyboard)
                    return { clicked: false, needsEscape: true };
                }
                
                return dismissModal();
                """

                dismiss_result = await self.page.evaluate(dismiss_script)

                # If JavaScript couldn't click anything but suggests Escape key
                if not dismiss_result.get("clicked", False) and dismiss_result.get("needsEscape", False):
                    await self.page.keyboard.press("Escape")
                    return {"success": True, "message": "Pressed Escape key to dismiss modal", "action_taken": "escape_key"}
                elif dismiss_result.get("clicked", False):
                    return {"success": True, "message": f"Clicked {dismiss_result.get('element')} to dismiss modal", "action_taken": "click_dismiss"}

            elif "accept_cookies" in recovery_actions:
                logger.info("Recovery action: Attempting to accept cookies")
                # Try to accept cookies by clicking common cookie consent buttons
                cookie_script = """
                function acceptCookies() {
                    // Common cookie acceptance button texts
                    const acceptPatterns = [
                        'accept', 'agree', 'ok', 'got it', 'i understand', 
                        'allow', 'continue', 'consent', 'allow all', 'accept all'
                    ];
                    
                    // Try to find buttons with these texts
                    for (const pattern of acceptPatterns) {
                        // Look for buttons
                        const buttons = Array.from(document.querySelectorAll('button')).filter(
                            el => el.innerText.toLowerCase().includes(pattern)
                        );
                        
                        // Also check for links and other clickable elements
                        const links = Array.from(document.querySelectorAll('a')).filter(
                            el => el.innerText.toLowerCase().includes(pattern)
                        );
                        
                        const otherElements = Array.from(document.querySelectorAll(
                            '[role="button"], .btn, .button'
                        )).filter(
                            el => el.innerText.toLowerCase().includes(pattern)
                        );
                        
                        const allElements = [...buttons, ...links, ...otherElements];
                        
                        // Filter to only visible elements
                        const visibleElements = allElements.filter(el => {
                            const rect = el.getBoundingClientRect();
                            const style = window.getComputedStyle(el);
                            return style.display !== 'none' && 
                                   style.visibility !== 'hidden' &&
                                   rect.width > 0 &&
                                   rect.height > 0 &&
                                   rect.top > 0 &&
                                   rect.top < window.innerHeight;
                        });
                        
                        if (visibleElements.length > 0) {
                            visibleElements[0].click();
                            return { clicked: true, text: visibleElements[0].innerText, pattern: pattern };
                        }
                    }
                    
                    return { clicked: false };
                }
                
                return acceptCookies();
                """

                cookie_result = await self.page.evaluate(cookie_script)

                if cookie_result.get("clicked", False):
                    return {
                        "success": True,
                        "message": f"Accepted cookies by clicking '{cookie_result.get('text')}'",
                        "action_taken": "accept_cookies"
                    }

            # For other error types like authentication or CAPTCHA, we can't auto-recover
            return {"success": False, "message": f"No automatic recovery available for {error_type}", "action_taken": "none"}

        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}", exc_info=True)
            return {"success": False, "message": f"Recovery failed: {str(e)}", "action_taken": "failed"}

    async def capture_screenshot(self) -> Dict[str, Any]:
        """Capture a screenshot of the current page state for visual verification"""
        if not self.page:
            return {"success": False, "message": "Browser not initialized"}

        try:
            # Capture screenshot as bytes
            screenshot_bytes = await self.page.screenshot(type="jpeg", quality=75)

            # Convert to base64 for transmission over WebSocket
            import base64
            base64_screenshot = base64.b64encode(
                screenshot_bytes).decode("utf-8")

            return {
                "success": True,
                "message": "Screenshot captured successfully",
                "screenshot": f"data:image/jpeg;base64,{base64_screenshot}"
            }
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}", exc_info=True)
            return {"success": False, "message": f"Screenshot capture failed: {str(e)}"}

    async def compare_visual_states(self, before_screenshot: str, after_screenshot: str) -> Dict[str, Any]:
        """Compare two screenshots to determine if there was a visual change
        This can be used to verify that an action had a visible effect"""
        if not before_screenshot or not after_screenshot:
            return {"success": False, "message": "Cannot compare screenshots: missing data"}

        try:
            # Extract base64 data
            before_data = before_screenshot.split(
                ",")[1] if "," in before_screenshot else before_screenshot
            after_data = after_screenshot.split(
                ",")[1] if "," in after_screenshot else after_screenshot

            import base64
            from PIL import Image
            import io
            import numpy as np

            # Convert base64 to images
            before_img = Image.open(io.BytesIO(base64.b64decode(before_data)))
            after_img = Image.open(io.BytesIO(base64.b64decode(after_data)))

            # Convert to numpy arrays for comparison
            before_array = np.array(before_img)
            after_array = np.array(after_img)

            # Handle different sizes
            if before_array.shape != after_array.shape:
                return {
                    "success": True,
                    "visual_change": True,
                    "message": "Visual state changed (different dimensions)",
                    "difference_score": 1.0
                }

            # Calculate difference
            difference = np.sum(
                np.abs(before_array - after_array)) / (before_array.size * 255)

            # Determine if there was a significant change
            # Threshold can be adjusted based on testing
            change_threshold = 0.01  # 1% pixel difference is considered significant
            visual_change = difference > change_threshold

            return {
                "success": True,
                "visual_change": visual_change,
                "difference_score": float(difference),
                "message": f"Visual comparison completed: {'Change detected' if visual_change else 'No significant change'}"
            }
        except Exception as e:
            logger.error(f"Error comparing visual states: {e}", exc_info=True)
            return {"success": False, "message": f"Visual comparison failed: {str(e)}"}

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
