import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from playwright.async_api import async_playwright

from browser import Browser

logger = logging.getLogger("web_automation")


class Agent:
    def __init__(self, api_key: str = None):
        self.browser = Browser()
        self.current_plan = []
        self.current_step_index = 0
        self.is_running = False
        self.is_paused = False
        self.is_waiting_user = False
        self.websocket = None
        self.task = None
        self.failed_action = None
        self.last_error_message = None  # Add field to store last error

        # Initialize LLM if API key provided
        if api_key:
            self.setup_llm(api_key)
        else:
            self.gemini_model = None

    def setup_llm(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash")
            logger.info("LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.gemini_model = None

    async def analyze_state(self, page_details: Dict[str, Any], step: str) -> Dict[str, Any]:
        """Analyze current browser state and determine the next action using a Playwright locator strategy."""
        if not self.gemini_model:
            return {"success": False, "message": "LLM not initialized"}

        # Store for potential logging on failure
        html_content = page_details.get('html_content', '')
        try:
            # Limit HTML content length again just before sending to LLM as a final safeguard
            MAX_PROMPT_HTML_LENGTH = 18000
            prompt_html_content = html_content
            if len(html_content) > MAX_PROMPT_HTML_LENGTH:
                prompt_html_content = html_content[:MAX_PROMPT_HTML_LENGTH] + \
                    "\n... (truncated for prompt)"

            prompt = f"""
            You are an expert web automation assistant using Playwright. Analyze the HTML and current step to determine the single best action, prioritizing robust Playwright locators.

            CURRENT PLAN STEP: {step}

            CURRENT WEB PAGE DETAILS:
            URL: {page_details.get('url')}
            TITLE: {page_details.get('title')}

            CURRENT PAGE HTML (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            1.  Review the STEP and HTML.
            2.  Choose ONE action: `navigate`, `click`, `type`, `wait`, `complete`.
            3.  **If choosing `click` or `type`:**
                a.  Identify the *exact* target element in the HTML.
                b.  Choose the **most appropriate Playwright locator strategy** based on the element's attributes in the HTML.
                c.  **Locator Strategy Priority:**
                    1.  `get_by_role`: If the element has a clear ARIA role (button, link, textbox, etc.) and accessible name (visible text, aria-label). Provide `role` and `name` (use `exact=True` for name if appropriate).
                    2.  `get_by_text`: If the element is uniquely identified by its exact visible text content. Provide `text` and `exact=True`.
                    3.  `get_by_label`: If the element is an input associated with a `<label>`. Provide the label `text`.
                    4.  `get_by_placeholder`: If the element has a unique `placeholder` attribute. Provide `placeholder` text.
                    5.  `get_by_test_id`: If the element has a `data-testid` attribute. Provide the `test_id`.
                    6.  `css`: As a last resort, if no other strategy fits well. Provide a specific, stable `selector` (prefer IDs, unique attributes like `name`, avoid dynamic classes).
                d.  Provide the chosen `strategy` and its corresponding arguments (`role`, `name`, `text`, `label`, `placeholder`, `test_id`, `selector`, `exact`).
            4.  If the step is done, choose `complete`. If waiting is needed, choose `wait`.

            **Output Format (JSON only):**
            Return ONLY a valid JSON object.
            ```json
            {{
                "action_type": "navigate|click|type|wait|complete",
                "locator_strategy": "css|get_by_role|get_by_text|get_by_label|get_by_placeholder|get_by_test_id|null",
                "locator_args": {{
                    "selector": "css selector string", // Only if strategy is 'css'
                    "role": "aria role string", // Only if strategy is 'get_by_role'
                    "name": "accessible name string", // Only if strategy is 'get_by_role'
                    "text": "text content string", // Only if strategy is 'get_by_text' or 'get_by_label'
                    "placeholder": "placeholder string", // Only if strategy is 'get_by_placeholder'
                    "test_id": "data-testid string", // Only if strategy is 'get_by_test_id'
                    "exact": true | false // Optional, for get_by_role, get_by_text
                    // Add other potential args like 'level' for heading if needed
                }}, // Arguments for the chosen strategy, null if not applicable
                "value": "URL, text to type, or wait duration (null if not applicable)",
                "explanation": "Brief justification for the chosen strategy and arguments, referencing HTML evidence."
            }}
            """
            # Run the LLM in a threadpool to not block
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.05,  # Keep low for precision
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            # Clean potential markdown formatting
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            try:
                action = json.loads(text.strip())
                # Basic validation
                if isinstance(action, dict) and action.get("action_type"):
                    # Further validation for locator actions
                    if action["action_type"] in ["click", "type"]:
                        if not action.get("locator_strategy") or not action.get("locator_args"):
                            logger.error(
                                f"LLM returned click/type action without locator strategy/args: {action}")
                            # Return HTML used for analysis, useful for debugging _handle_failure
                            return {"success": False, "message": "LLM returned invalid locator details.", "html_used": html_content}
                    logger.info(f"LLM action determined: {action}")
                    # Return HTML used for analysis, useful for debugging _handle_failure
                    return {"success": True, "action": action, "html_used": html_content}
                else:
                    logger.error(
                        f"LLM returned invalid action structure: {action}")
                    # Return HTML used for analysis, useful for debugging _handle_failure
                    return {"success": False, "message": "LLM returned invalid action structure.", "html_used": html_content}
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode LLM JSON response: {e}. Response text: {text}")
                # Return HTML used for analysis, useful for debugging _handle_failure
                return {"success": False, "message": f"Failed to decode LLM JSON response: {e}", "html_used": html_content}

        except Exception as e:
            # Log full traceback
            logger.error(f"Error analyzing state: {e}", exc_info=True)
            # Return HTML used for analysis, useful for debugging _handle_failure
            return {"success": False, "message": f"Analysis failed: {e}", "html_used": html_content}

    async def create_plan(self, task: str, initial_page_details: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate a step-by-step plan for the task, considering the initial page state."""
        if not self.gemini_model:
            logger.warning("LLM not available, returning default plan.")
            # Fallback
            return ["Navigate to website", "Perform actions", "Complete task"]

        state_context = "No initial page state provided. Assume starting from a blank page."
        if initial_page_details and initial_page_details.get("success"):
            # Limit HTML context for the planning prompt
            html_content = initial_page_details.get("html_content", "")
            MAX_PLAN_HTML_LENGTH = 4000  # Keep planning prompt concise
            if len(html_content) > MAX_PLAN_HTML_LENGTH:
                html_content = html_content[:MAX_PLAN_HTML_LENGTH] + \
                    "\n... (truncated)"

            state_context = f"""
            CURRENT WEB PAGE STATE:
            URL: {initial_page_details.get("url", "unknown")}
            TITLE: {initial_page_details.get("title", "unknown")}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {html_content}
            ```
            """

        try:
            planning_prompt = f"""
            You are a web automation planner. Create a concise, actionable, step-by-step plan (3-7 steps) to achieve the user's task, starting from the current web page state provided below.

            USER TASK: {task}

            {state_context}

            **Instructions:**
            - Each step should represent a distinct, high-level user action (e.g., "Log in", "Search for 'product name'", "Add item to cart", "Fill out shipping form").
            - Ensure the plan logically progresses towards the user's task *from the current state*.
            - If the initial state suggests the task is partially complete or on an unexpected page, adjust the plan accordingly (e.g., navigate first if needed).
            - Be specific where possible (e.g., mention what to search for).

            **Output Format (JSON only):**
            Return ONLY a JSON array of strings representing the plan steps. Ensure the JSON is valid.
            Example: ["Navigate to login page", "Enter username and password", "Click login button", "Verify login success"]
            """

            # Run the LLM in a threadpool to not block
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                planning_prompt,
                generation_config={
                    "temperature": 0.3,  # Slightly higher temp for planning creativity
                    "response_mime_type": "application/json"
                }
            )

            # Parse response
            text = response.text
            # Clean potential markdown formatting (same as analyze_state)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            try:
                steps = json.loads(text.strip())
                if isinstance(steps, list) and all(isinstance(step, str) for step in steps) and steps:
                    logger.info(
                        f"LLM generated plan considering initial state: {steps}")
                    return steps
                else:
                    logger.error(
                        f"LLM plan generation failed to return a valid non-empty list: {steps}")
                    # Fallback
                    return ["Navigate to website", "Perform actions", "Complete task"]
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode LLM JSON plan response: {e}. Response text: {text}")
                # Fallback
                return ["Navigate to website", "Perform actions", "Complete task"]

        except Exception as e:
            logger.error(f"Failed to create plan: {e}", exc_info=True)
            # Fallback
            return ["Navigate to website", "Perform actions", "Complete task"]

    async def start_execution(self, plan: List[str], websocket) -> bool:
        """Start executing the provided plan"""
        if self.is_running:
            return False

        self.current_plan = plan
        self.current_step_index = 0
        self.is_running = True
        self.is_paused = False
        self.is_waiting_user = False
        self.websocket = websocket

        # Initialize browser
        success = await self.browser.initialize()
        if not success:
            await self._send_message("error", "Failed to initialize browser")
            self.is_running = False
            return False

        # Start execution as a task
        self.task = asyncio.create_task(self._execution_loop())
        return True

    async def _send_browser_screenshot(self):
        """Helper to get and send screenshot"""
        screenshot = await self.browser.get_screenshot()
        if screenshot:
            await self._send_message("browser_screenshot", {"screenshot": screenshot})

    async def _execution_loop(self):
        """Main execution loop with dynamic plan re-evaluation and screenshot updates."""
        logger.info("Starting execution loop")
        last_page_details_analyzed = None

        try:
            # Send initial screenshot if browser is ready
            if self.browser.page and not self.browser.page.is_closed():
                await self._send_browser_screenshot()

            while self.is_running and self.current_step_index < len(self.current_plan):
                if self.is_paused:
                    await asyncio.sleep(0.5)
                    continue

                current_step = self.current_plan[self.current_step_index]
                await self._send_message("executing_step", {
                    "step_index": self.current_step_index,
                    "step": current_step
                })

                # Get current page state (using get_page_details for full HTML)
                page_details = await self.browser.get_page_details()
                if not page_details.get("success", False):
                    await self._handle_failure(f"Failed to get page details: {page_details.get('message')}", None, page_details)
                    continue

                # Send screenshot before analysis
                await self._send_browser_screenshot()

                last_page_details_analyzed = page_details

                # Analyze state and determine action
                await self._send_message("analyzing", {
                    "message": f"Analyzing page for step: {current_step}"
                })
                analysis = await self.analyze_state(page_details, current_step)

                if not analysis.get("success", False):
                    await self._handle_failure(
                        f"Analysis failed: {analysis.get('message')}", None, last_page_details_analyzed)
                    continue

                action = analysis.get("action", {})
                action_type = action.get("action_type")

                # Execute the determined action
                await self._send_message("action", {
                    "type": action_type,
                    "details": action
                })
                result = await self._execute_action(action)

                # Send screenshot *after* action attempt (success or fail)
                await self._send_browser_screenshot()

                if not result.get("success", False):
                    await self._handle_failure(
                        f"Action failed: {result.get('message')}", action, last_page_details_analyzed)
                    continue

                # --- Action Successful ---
                await self._send_message("step_completed", {
                    "step_index": self.current_step_index,
                    "message": f"Completed: {current_step}"
                })
                last_page_details_analyzed = None

                # --- Dynamic Plan Re-evaluation ---
                next_step_index = self.current_step_index + 1
                if next_step_index < len(self.current_plan):
                    logger.info("Re-evaluating remaining plan steps...")
                    # Get state after successful action
                    current_page_details = await self.browser.get_page_details()
                    if current_page_details.get("success"):
                        original_remaining_steps = self.current_plan[next_step_index:]
                        new_remaining_steps = await self.reevaluate_plan(current_page_details, original_remaining_steps)
                        if new_remaining_steps is not None and new_remaining_steps != original_remaining_steps:
                            logger.info(
                                f"Plan adjusted. Original remaining: {original_remaining_steps}, New remaining: {new_remaining_steps}")
                            completed_steps = self.current_plan[:next_step_index]
                            self.current_plan = completed_steps + new_remaining_steps
                            await self._send_message("plan_created", {"plan": self.current_plan})
                        elif new_remaining_steps is None:
                            logger.warning(
                                "Plan re-evaluation failed, continuing with original plan.")
                        else:
                            logger.info(
                                "Plan re-evaluation confirmed remaining steps are still valid.")
                    else:
                        logger.warning(
                            "Could not get page details for plan re-evaluation, continuing with original plan.")

                # Move to next step
                self.current_step_index += 1
                await asyncio.sleep(1)

            # Execution complete
            if self.is_running and self.current_step_index >= len(self.current_plan):
                await self._send_message("execution_complete", {
                    "message": "Plan execution completed successfully"
                })

        except Exception as e:
            logger.error(f"Execution error in loop: {e}", exc_info=True)
            await self._handle_failure(f"Unexpected error: {str(e)}", None, last_page_details_analyzed)

        finally:
            self.is_running = False

    async def reevaluate_plan(self, current_page_details: Dict[str, Any], remaining_steps: List[str]) -> Optional[List[str]]:
        """Check if the remaining plan steps are still valid given the current state, suggest minor adjustments if needed."""
        if not self.gemini_model or not remaining_steps:
            return remaining_steps  # No LLM or no steps left, return original

        try:
            # Limit HTML context for re-evaluation prompt
            html_content = current_page_details.get("html_content", "")
            MAX_REEVAL_HTML_LENGTH = 4000  # Keep prompt concise
            if len(html_content) > MAX_REEVAL_HTML_LENGTH:
                html_content = html_content[:MAX_REEVAL_HTML_LENGTH] + \
                    "\n... (truncated)"

            # Focus on the next few steps for efficiency
            steps_to_evaluate = remaining_steps[:3]  # Check next 1-3 steps

            prompt = f"""
            You are a web automation plan verifier. A step was just completed successfully. Based on the current page state, verify if the *next few* planned steps are still the most logical way to proceed towards the overall goal implied by the remaining steps.

            CURRENT WEB PAGE STATE:
            URL: {current_page_details.get("url", "unknown")}
            TITLE: {current_page_details.get("title", "unknown")}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {html_content}
            ```

            NEXT FEW PLANNED STEPS: {steps_to_evaluate}
            FULL REMAINING PLAN: {remaining_steps}

            **Instructions:**
            1. Analyze the CURRENT STATE and the NEXT FEW PLANNED STEPS.
            2. **If the NEXT FEW PLANNED STEPS are still appropriate and logical** given the current state, return the original FULL REMAINING PLAN exactly as provided.
            3. **If minor adjustments are needed** to the *upcoming* steps based on the current state (e.g., slightly different wording, combining/splitting the immediate next step), return a *new* list representing the adjusted FULL REMaining PLAN. Make minimal changes, focusing only on necessary corrections for the immediate future. Do NOT add completely new, unrelated steps.
            4. Ensure the returned list represents the *entire* sequence of steps from the next one onwards.

            **Output Format (JSON only):**
            Return ONLY a JSON array of strings representing the full remaining plan (either original or adjusted). Ensure the JSON is valid.
            Example Input FULL REMAINING PLAN: ["Click 'Next'", "Fill address", "Submit form"]
            Example Output (if valid): ["Click 'Next'", "Fill address", "Submit form"]
            Example Output (if 'Next' isn't visible but 'Continue' is): ["Click 'Continue'", "Fill address", "Submit form"]
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for verification/minor adjustment
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            # Clean potential markdown formatting
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            try:
                adjusted_steps = json.loads(text.strip())
                if isinstance(adjusted_steps, list) and all(isinstance(step, str) for step in adjusted_steps):
                    # Basic check: if the list is empty when original wasn't, it's likely an error
                    if not adjusted_steps and remaining_steps:
                        logger.warning(
                            "LLM re-evaluation returned empty list unexpectedly. Keeping original plan.")
                        return remaining_steps
                    return adjusted_steps
                else:
                    logger.error(
                        f"LLM plan re-evaluation failed to return a valid list: {adjusted_steps}")
                    return None  # Indicate failure
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode LLM JSON re-evaluation response: {e}. Response text: {text}")
                return None  # Indicate failure

        except Exception as e:
            logger.error(
                f"Error during plan re-evaluation: {e}", exc_info=True)
            return None  # Indicate failure

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action based on type and locator strategy."""
        action_type = action.get("action_type", "").lower()
        strategy = action.get("locator_strategy")
        args = action.get("locator_args", {})
        value = action.get("value")
        max_retries = 1  # Try the action once, then retry once more if it fails
        retry_delay = 1.5  # Seconds to wait before retrying
        timeout = 15000  # Use the increased timeout from browser.py

        if not self.browser.page:
            return {"success": False, "message": "Browser not initialized"}
        page = self.browser.page  # Get page object

        for attempt in range(max_retries + 1):
            locator = None
            locator_description = "N/A"  # For logging
            try:
                result = {}
                if action_type == "navigate":
                    if not value:
                        return {"success": False, "message": "Navigate action requires a URL in 'value'."}
                    # Use the existing browser method for navigation consistency
                    result = await self.browser.navigate(value)
                elif action_type in ["click", "type"]:
                    if not strategy or not args:
                        return {"success": False, "message": f"{action_type.capitalize()} action requires locator strategy and args."}

                    # --- Build Locator based on strategy ---
                    # Common optional arg, defaults to None (usually False)
                    exact = args.get("exact")

                    if strategy == "css":
                        selector = args.get("selector")
                        if not selector:
                            return {"success": False, "message": "CSS strategy requires 'selector' arg."}
                        locator = page.locator(selector)
                        locator_description = f"CSS={selector}"
                    elif strategy == "get_by_role":
                        role = args.get("role")
                        name = args.get("name")
                        if not role:
                            return {"success": False, "message": "GetByRole strategy requires 'role' arg."}
                        # Build kwargs dict carefully, only including args that are not None
                        locator_kwargs = {}
                        if name is not None:
                            locator_kwargs["name"] = name
                        if exact is not None:
                            locator_kwargs["exact"] = exact
                        locator = page.get_by_role(role, **locator_kwargs)
                        locator_description = f"Role={role}, Args={locator_kwargs}"
                    elif strategy == "get_by_text":
                        text = args.get("text")
                        if text is None:
                            return {"success": False, "message": "GetByText strategy requires 'text' arg."}
                        locator = page.get_by_text(text, exact=exact)
                        locator_description = f"Text='{text}', Exact={exact}"
                    elif strategy == "get_by_label":
                        # Assuming label text is passed in 'text'
                        label_text = args.get("text")
                        if label_text is None:
                            return {"success": False, "message": "GetByLabel strategy requires label 'text' arg."}
                        locator = page.get_by_label(label_text, exact=exact)
                        locator_description = f"Label='{label_text}', Exact={exact}"
                    elif strategy == "get_by_placeholder":
                        placeholder = args.get("placeholder")
                        if placeholder is None:
                            return {"success": False, "message": "GetByPlaceholder strategy requires 'placeholder' arg."}
                        locator = page.get_by_placeholder(
                            placeholder, exact=exact)
                        locator_description = f"Placeholder='{placeholder}', Exact={exact}"
                    elif strategy == "get_by_test_id":
                        test_id = args.get("test_id")
                        if test_id is None:
                            return {"success": False, "message": "GetByTestId strategy requires 'test_id' arg."}
                        locator = page.get_by_test_id(test_id)
                        locator_description = f"TestId='{test_id}'"
                    else:
                        return {"success": False, "message": f"Unsupported locator strategy: {strategy}"}

                    # --- Perform Action using Locator ---
                    # Wait for the element to be visible using the chosen locator
                    await locator.wait_for(state="visible", timeout=timeout)

                    if action_type == "click":
                        # Shorter timeout for the click itself after visibility confirmed
                        await locator.click(timeout=5000)
                        result = {
                            "success": True, "message": f"Clicked element found by {locator_description}"}
                    elif action_type == "type":
                        if value is None:
                            return {"success": False, "message": "Type action requires a 'value'."}
                        # Shorter timeout for fill after visibility confirmed
                        await locator.fill(value, timeout=5000)
                        result = {
                            "success": True, "message": f"Typed '{value}' into element found by {locator_description}"}

                elif action_type == "wait":
                    try:
                        seconds = float(value if value is not None else 1)
                        await asyncio.sleep(seconds)
                        result = {"success": True,
                                  "message": f"Waited for {seconds} seconds"}
                    except (ValueError, TypeError):
                        await asyncio.sleep(1)
                        result = {"success": True,
                                  "message": "Waited for 1 second (default)"}
                elif action_type == "complete":
                    result = {"success": True,
                              "message": "Complete action"}
                else:
                    # Handle unknown action type explicitly
                    result = {
                        "success": False, "message": f"Unknown or unsupported action type: {action_type}"}

                # --- Handle Result ---
                if result.get("success", False):
                    return result
                # If max retries reached, format the error message before returning
                elif attempt == max_retries:
                    # Include locator description in final error if applicable
                    error_prefix = f"Action '{action_type}'"
                    if locator_description != "N/A":
                        error_prefix += f" using {locator_description}"
                    result["message"] = f"{error_prefix} failed. Error: {result.get('message')}"
                    return result
                # Log warning and retry if not max retries
                else:
                    logger.warning(
                        f"Action '{action_type}' using {locator_description} failed on attempt {attempt + 1}. Retrying in {retry_delay}s... Error: {result.get('message')}")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                # Catch exceptions during locator building or action execution
                error_msg = f"Exception during action '{action_type}'"
                if locator_description != "N/A":
                    error_msg += f" using {locator_description}"
                error_msg += f" (Attempt {attempt + 1}): {str(e)}"

                logger.error(error_msg, exc_info=True)  # Log with traceback
                if attempt == max_retries:
                    return {"success": False, "message": error_msg}
                else:
                    logger.warning(
                        f"Retrying action '{action_type}' after exception...")
                    await asyncio.sleep(retry_delay)

        # This should theoretically not be reached if max_retries >= 0, but as a fallback:
        return {"success": False, "message": f"Action '{action_type}' failed using {locator_description} after {max_retries + 1} attempts."}

    async def _handle_failure(self, error_message: str, failed_action: Optional[Dict], page_details: Optional[Dict] = None):
        """Handle action failure by pausing execution and storing error context."""
        logger.error(f"Execution failed: {error_message}")
        self.is_paused = True
        self.is_waiting_user = True
        self.failed_action = failed_action
        self.last_error_message = error_message  # Store the error message

        await self._send_message("execution_paused", {
            "message": error_message,
            "step_index": self.current_step_index,
            "failed_action": failed_action,
            "page_details": page_details
        })

    async def _send_message(self, msg_type: str, data: Any = None):
        """Send message via websocket if available"""
        if not self.websocket:
            return

        try:
            payload = {"type": msg_type}
            if isinstance(data, dict):
                payload.update(data)
            elif data is not None:
                payload["data"] = data

            await self.websocket.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def resume_execution(self):
        """Resumes execution if paused."""
        logger.debug("Resume execution called.")  # Added log
        if not self.is_running:
            logger.warning("Cannot resume, execution is not running.")
            return False
        if not self.is_paused:
            logger.warning("Cannot resume, execution is not paused.")
            return False

        logger.info("Resuming execution.")
        self.is_paused = False
        self.is_waiting_user = False
        self.last_error_message = None  # Clear last error on resume
        await self._send_message(self.websocket, "resuming", {
            "message": "Resuming execution",
            "step_index": self.current_step_index
        })
        logger.debug("Resume state updated and message sent.")  # Added log
        return True

    async def skip_current_step(self):
        """Skip the current step and move to the next one"""
        if not self.is_paused or self.current_step_index >= len(self.current_plan) - 1:
            return False

        self.current_step_index += 1
        await self._send_message("step_skipped", {
            "message": "Skipped to next step",
            "step_index": self.current_step_index
        })
        return True

    async def stop_execution(self):
        """Stop the execution completely"""
        self.is_running = False
        self.is_paused = False
        await self._send_message("execution_stopped", {
            "message": "Execution stopped by user"
        })

        # Cancel task if running
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        return True
