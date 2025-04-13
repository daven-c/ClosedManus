import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from playwright.async_api import async_playwright

from browser import Browser

# General logger
logger = logging.getLogger("web_automation")

# --- Dedicated logger for agent thoughts ---
agent_thoughts_logger = logging.getLogger("agent_thoughts")
agent_thoughts_logger.setLevel(logging.DEBUG)  # Log all thoughts
# Prevent thoughts from propagating to the root logger/console
agent_thoughts_logger.propagate = False
# Create file handler
try:
    # Use 'a' for append mode
    fh = logging.FileHandler("web_agents.log", mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the handler to the logger
    if not agent_thoughts_logger.hasHandlers():  # Avoid adding multiple handlers on reload
        agent_thoughts_logger.addHandler(fh)
except Exception as e:
    logger.error(f"Failed to set up agent_thoughts logger: {e}")
# --- End dedicated logger setup ---


class Agent:
    def __init__(self, api_key: str = None):
        self.browser = Browser()
        self.goal: Optional[str] = None
        self.completed_steps: List[str] = []
        self.is_running = False
        self.is_paused = False
        self.is_waiting_user = False
        self.websocket = None
        self.task = None
        self.failed_action: Optional[Dict] = None
        self.last_error_message: Optional[str] = None
        self.last_step_description: Optional[str] = None

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
        if not self.gemini_model:
            return {"success": False, "message": "LLM not initialized"}

        html_content = page_details.get('html_content', '')
        try:
            # Log length received from browser.py
            received_length = len(html_content)
            logger.debug(
                f"analyze_state received HTML length: {received_length}")

            MAX_PROMPT_HTML_LENGTH = 18000
            prompt_html_content = html_content
            final_prompt_length = received_length
            if received_length > MAX_PROMPT_HTML_LENGTH:
                prompt_html_content = html_content[:MAX_PROMPT_HTML_LENGTH] + \
                    "\n... (truncated for analyze_state prompt)"
                final_prompt_length = len(prompt_html_content)
                logger.warning(
                    f"analyze_state truncating HTML from {received_length} to {final_prompt_length} for prompt.")
            else:
                logger.debug(
                    f"analyze_state using HTML length {final_prompt_length} for prompt.")

            prompt = f"""
            You are an expert web automation assistant using Playwright. Analyze the HTML and current step description to determine the single best action, prioritizing robust Playwright locators.

            CURRENT STEP DESCRIPTION: {step}

            CURRENT WEB PAGE DETAILS:
            URL: {page_details.get('url')}
            TITLE: {page_details.get('title')}

            CURRENT PAGE HTML (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            1.  Review the STEP DESCRIPTION and HTML.
            2.  Choose ONE action: `navigate`, `click`, `type`, `wait`, `complete`, `javascript` (use 'complete' if the step description itself is already satisfied by the current state).
            3.  **If choosing `click` or `type`:**
                a.  Identify the *exact* target element in the HTML relevant to the step description.
                b.  Choose the **most appropriate Playwright locator strategy** based on the element's attributes in the HTML.
                c.  **Locator Strategy Priority:** `get_by_role`, `get_by_text`, `get_by_label`, `get_by_placeholder`, `get_by_test_id`, `css` (last resort).
                d.  Provide the chosen `strategy` and its corresponding arguments (`role`, `name`, `text`, `label`, `placeholder`, `test_id`, `selector`, `exact`).
            4.  **If choosing `javascript`:**
                a.  Write a concise JavaScript snippet that interacts with the page to achieve the step.
                b.  The script should handle element selection and interaction directly in the browser context.
                c.  You can use document.querySelector, document.querySelectorAll and other DOM APIs to find and interact with elements.
                d.  Your script can return values back to the agent if needed.
            5.  If waiting is needed, choose `wait`.

            **Output Format (JSON only):**
            Return ONLY a valid JSON object. Include a brief 'thought' process explaining your choices.
            ```json
            {{
                "action_type": "navigate|click|type|wait|complete|javascript",
                "locator_strategy": "css|get_by_role|get_by_text|get_by_label|get_by_placeholder|get_by_test_id|null",
                "locator_args": {{ ... arguments ... }},
                "value": "URL, text to type, wait duration, or null if not applicable",
                "javascript_code": "JavaScript code to execute in browser context (only for javascript action type)",
                "explanation": "Brief justification for the chosen strategy and arguments, referencing HTML evidence.",
                "thought": "Your reasoning for choosing this action, strategy, and arguments based on the step and HTML."
            }}
            """
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.05,
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            try:
                action = json.loads(text.strip())
                if isinstance(action, dict) and action.get("action_type"):
                    thought = action.get("thought", "No thought provided.")
                    agent_thoughts_logger.info(
                        f"Analysis for '{step}': {thought}")
                    await self._send_message("agent_thought", {"thought": f"Analysis for '{step}': {thought}"})

                    if action["action_type"] in ["click", "type"]:
                        if not action.get("locator_strategy") or not action.get("locator_args"):
                            logger.error(
                                f"LLM returned click/type action without locator strategy/args: {action}")
                            return {"success": False, "message": "LLM returned invalid locator details.", "html_used": html_content}
                    elif action["action_type"] == "javascript":
                        if not action.get("javascript_code"):
                            logger.error(
                                f"LLM returned javascript action without code: {action}")
                            return {"success": False, "message": "LLM returned javascript action without code.", "html_used": html_content}
                    logger.info(f"LLM action determined: {action}")
                    return {"success": True, "action": action, "html_used": html_content}
                else:
                    logger.error(
                        f"LLM returned invalid action structure: {action}")
                    return {"success": False, "message": "LLM returned invalid action structure.", "html_used": html_content}
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode LLM JSON response: {e}. Response text: {text}")
                return {"success": False, "message": f"Failed to decode LLM JSON response: {e}", "html_used": html_content}

        except Exception as e:
            logger.error(f"Error analyzing state: {e}", exc_info=True)
            return {"success": False, "message": f"Analysis failed: {e}", "html_used": html_content}

    async def analyze_possible_actions(self, page_details: Dict[str, Any], step_description: str) -> Dict[str, Any]:
        """Scans the page using JavaScript first, then determines the best action to take based on the scan results"""
        if not self.gemini_model:
            return {"success": False, "message": "LLM not initialized"}

        # First, scan the page for actionable elements using JavaScript
        scan_result = await self.browser.scan_actionable_elements()
        if not scan_result.get("success", False):
            logger.warning(
                f"Element scanning failed: {scan_result.get('message')}")
            # Fall back to regular HTML analysis if JavaScript scan fails
            return await self.analyze_state(page_details, step_description)

        html_content = page_details.get('html_content', '')
        actionable_elements = scan_result.get("scan_result", {})

        try:
            # Truncate HTML if needed for prompt size
            MAX_PROMPT_HTML_LENGTH = 6000  # Reduced size since we're also including scan results
            prompt_html_content = html_content
            received_length = len(html_content)

            if received_length > MAX_PROMPT_HTML_LENGTH:
                prompt_html_content = html_content[:MAX_PROMPT_HTML_LENGTH] + \
                    "\n... (truncated for action determination prompt)"
                logger.debug(
                    f"Truncating HTML from {received_length} to {MAX_PROMPT_HTML_LENGTH} for action prompt.")

            # Convert actionable elements scan to a structured format for the prompt
            elements_json = json.dumps(actionable_elements, indent=2)

            prompt = f"""
            You are an expert web automation assistant using Playwright. Analyze the page scan data and current step description to determine the best action.

            CURRENT STEP DESCRIPTION: {step_description}

            CURRENT WEB PAGE DETAILS:
            URL: {page_details.get('url')}
            TITLE: {page_details.get('title')}

            JAVASCRIPT SCAN RESULTS - ACTIONABLE ELEMENTS:
            ```json
            {elements_json}
            ```

            PAGE HTML EXCERPT (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            1.  Review the STEP DESCRIPTION, JAVASCRIPT SCAN RESULTS, and HTML.
            2.  Based on the JavaScript scan of actionable elements, choose ONE action: `navigate`, `click`, `type`, `wait`, `complete`, `javascript`.
            3.  Prioritize using the JavaScript scan results to select elements rather than parsing the HTML.
            4.  **If choosing `click`, create a precise JavaScript snippet to find and click the element.**
            5.  **If choosing `type`, create a precise JavaScript snippet to find the input field and enter text.**
            6.  For complex interactions, use the `javascript` action type with custom JavaScript code.

            **Output Format (JSON only):**
            Return ONLY a valid JSON object. Include a brief 'thought' process explaining your choices.
            ```json
            {{
                "action_type": "navigate|click|type|wait|complete|javascript",
                "javascript_code": "JavaScript code to execute (for click, type, and javascript action types)",
                "value": "URL to navigate to, text to type, or wait duration in seconds",
                "explanation": "Brief justification for the chosen action and elements",
                "thought": "Your reasoning for choosing this action based on the step and scan results"
            }}
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.05,
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            try:
                action = json.loads(text.strip())
                if isinstance(action, dict) and action.get("action_type"):
                    thought = action.get("thought", "No thought provided.")
                    agent_thoughts_logger.info(
                        f"JS-based analysis for '{step_description}': {thought}")
                    await self._send_message("agent_thought", {"thought": f"Analysis for '{step_description}': {thought}"})

                    # Ensure JavaScript code exists for interactive actions
                    if action["action_type"] in ["click", "type", "javascript"] and not action.get("javascript_code"):
                        logger.error(
                            f"LLM returned {action['action_type']} action without JavaScript code: {action}")
                        return {"success": False, "message": f"LLM returned {action['action_type']} action without JavaScript code.", "scan_result": scan_result}

                    # Ensure value exists for type and navigate actions
                    if action["action_type"] in ["navigate", "type"] and not action.get("value"):
                        logger.error(
                            f"LLM returned {action['action_type']} action without value: {action}")
                        return {"success": False, "message": f"LLM returned {action['action_type']} action without value.", "scan_result": scan_result}

                    logger.info(
                        f"LLM JavaScript-based action determined: {action}")
                    return {"success": True, "action": action, "scan_result": scan_result, "html_used": html_content}
                else:
                    logger.error(
                        f"LLM returned invalid action structure: {action}")
                    return {"success": False, "message": "LLM returned invalid action structure.", "scan_result": scan_result}
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode LLM JSON response: {e}. Response text: {text}")
                return {"success": False, "message": f"Failed to decode LLM JSON response: {e}", "scan_result": scan_result}

        except Exception as e:
            logger.error(
                f"Error analyzing possible actions: {e}", exc_info=True)
            return {"success": False, "message": f"Analysis failed: {e}", "scan_result": scan_result}

    async def plan_next_step(self, current_page_details: Dict[str, Any]) -> Optional[str]:
        if not self.gemini_model:
            logger.warning("LLM not available, cannot plan next step.")
            return None

        try:
            html_content = current_page_details.get("html_content", "")
            received_length = len(html_content)
            logger.debug(
                f"plan_next_step received HTML length: {received_length}")

            MAX_PLAN_HTML_LENGTH = 4000
            prompt_html_content = html_content
            final_prompt_length = received_length
            if received_length > MAX_PLAN_HTML_LENGTH:
                prompt_html_content = html_content[:MAX_PLAN_HTML_LENGTH] + \
                    "\n... (truncated for plan_next_step prompt)"
                final_prompt_length = len(prompt_html_content)
                logger.warning(
                    f"plan_next_step truncating HTML from {received_length} to {final_prompt_length} for prompt.")
            else:
                logger.debug(
                    f"plan_next_step using HTML length {final_prompt_length} for prompt.")

            history = "\n".join(f"- {s}" for s in self.completed_steps)
            if not history:
                history = "No steps completed yet."

            prompt = f"""
            You are a web automation planner determining the *single next step*.

            OVERALL GOAL: {self.goal}

            COMPLETED STEPS HISTORY:
            {history}

            CURRENT WEB PAGE STATE:
            URL: {current_page_details.get("url", "unknown")}
            TITLE: {current_page_details.get("title", "unknown")}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            Based on the OVERALL GOAL, the COMPLETED STEPS HISTORY, and the CURRENT WEB PAGE STATE, determine the **single most logical next step** to take.
            - The step should be a high-level action description (e.g., "Click the 'Login' button", "Enter 'password123' into the password field", "Navigate to https://example.com/cart").
            - Consider the history to avoid repeating actions unnecessarily.
            - Ensure the step is actionable given the current state.

            **Output Format (JSON only):**
            Return ONLY a JSON object containing the next step description and your reasoning.
            ```json
            {{
                "next_step": "Description of the single next step to perform.",
                "thought": "Your reasoning for choosing this step based on the goal, history, and current state."
            }}
            """
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                }
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            next_step = result.get("next_step")
            thought = result.get("thought", "No thought provided.")

            agent_thoughts_logger.info(f"Planning next step: {thought}")
            await self._send_message("agent_thought", {"thought": f"Planning next step: {thought}"})

            if next_step and isinstance(next_step, str):
                logger.info(f"LLM planned next step: {next_step}")
                return next_step
            else:
                logger.error(
                    f"LLM failed to provide a valid next step: {result}")
                return None
        except Exception as e:
            logger.error(f"Error planning next step: {e}", exc_info=True)
            return None

    async def check_goal_completion(self, current_page_details: Dict[str, Any]) -> bool:
        if not self.gemini_model:
            logger.warning("LLM not available, assuming goal not met.")
            return False

        try:
            html_content = current_page_details.get("html_content", "")
            received_length = len(html_content)
            logger.debug(
                f"check_goal_completion received HTML length: {received_length}")

            MAX_CHECK_HTML_LENGTH = 4000
            prompt_html_content = html_content
            final_prompt_length = received_length
            if received_length > MAX_CHECK_HTML_LENGTH:
                prompt_html_content = html_content[:MAX_CHECK_HTML_LENGTH] + \
                    "\n... (truncated for check_goal_completion prompt)"
                final_prompt_length = len(prompt_html_content)
                logger.warning(
                    f"check_goal_completion truncating HTML from {received_length} to {final_prompt_length} for prompt.")
            else:
                logger.debug(
                    f"check_goal_completion using HTML length {final_prompt_length} for prompt.")

            history = "\n".join(f"- {s}" for s in self.completed_steps)

            prompt = f"""
            You are a web automation goal checker.

            OVERALL GOAL: {self.goal}

            COMPLETED STEPS HISTORY:
            {history}

            CURRENT WEB PAGE STATE:
            URL: {current_page_details.get("url", "unknown")}
            TITLE: {current_page_details.get("title", "unknown")}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            Based *only* on the OVERALL GOAL, the COMPLETED STEPS HISTORY, and the CURRENT WEB PAGE STATE, determine if the overall goal has been successfully achieved.
            - Consider if the current URL, title, or key elements in the HTML indicate success.
            - Look for confirmation messages, expected final content, or redirection to a success page relevant to the goal.

            **Output Format (JSON only):**
            Return ONLY a JSON object with a boolean value and your reasoning.
            ```json
            {{
                "goal_achieved": true | false,
                "thought": "Your reasoning for concluding whether the goal is achieved based on the state and history."
            }}
            """
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            )
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            goal_achieved = result.get("goal_achieved", False)
            thought = result.get("thought", "No thought provided.")

            agent_thoughts_logger.info(f"Goal check: {thought}")
            await self._send_message("agent_thought", {"thought": f"Goal check: {thought}"})

            logger.info(f"LLM goal check result: {goal_achieved}")
            return goal_achieved
        except Exception as e:
            logger.error(f"Error checking goal completion: {e}", exc_info=True)
            return False

    async def create_multi_step_plan(self, current_page_details: Dict[str, Any]) -> Optional[List[str]]:
        """Creates a comprehensive multi-step plan and returns it as a list of steps.
        This provides lookahead planning capability similar to enterprise automation systems.
        """
        if not self.gemini_model:
            logger.warning("LLM not available, cannot create multi-step plan.")
            return None

        try:
            html_content = current_page_details.get("html_content", "")
            url = current_page_details.get("url", "unknown")
            title = current_page_details.get("title", "unknown")

            # Truncate HTML to avoid token limits
            MAX_PLAN_HTML_LENGTH = 10000
            prompt_html_content = html_content[:MAX_PLAN_HTML_LENGTH] if len(
                html_content) > MAX_PLAN_HTML_LENGTH else html_content

            history = "\n".join(f"- {s}" for s in self.completed_steps)
            if not history:
                history = "No steps completed yet."

            # First scan actionable elements to understand what's possible
            scan_result = await self.browser.scan_actionable_elements()
            elements_summary = "No actionable elements found on page (JavaScript scan failed)."
            if scan_result.get("success") and scan_result.get("scan_result"):
                # Summarize available actions to inform planning
                elements_data = scan_result["scan_result"]
                stats = elements_data.get("stats", {})
                elements_summary = f"Available elements: {stats.get('buttons', 0)} buttons, {stats.get('links', 0)} links, {stats.get('inputs', 0)} input fields, {stats.get('selects', 0)} dropdowns"

            prompt = f"""
            You are an expert web automation planning system. Create a detailed, multi-step plan to achieve the following goal:
            
            GOAL: {self.goal}
            
            CURRENT CONTEXT:
            - URL: {url}
            - Page title: {title}
            - {elements_summary}
            
            COMPLETED STEPS SO FAR:
            {history}
            
            INSTRUCTIONS:
            1. Create a comprehensive multi-step plan (3-7 steps) that will achieve the goal from the current state
            2. Each step should be clear, specific, and actionable (e.g., "Click the Submit button" rather than "Submit the form")
            3. Consider possible edge cases and handling for each step
            4. Ensure steps are in logical order and build toward the goal efficiently
            5. For each step, think about how its success can be verified after execution
            
            Return ONLY a JSON array of steps represented as strings.
            ```json
            [
              "Step 1 description",
              "Step 2 description",
              "etc..."
            ]
            ```
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed_steps = json.loads(text.strip())
            if isinstance(parsed_steps, list) and all(isinstance(step, str) for step in parsed_steps):
                logger.info(
                    f"Generated multi-step plan with {len(parsed_steps)} steps")

                # Log the generated plan
                for i, step in enumerate(parsed_steps):
                    agent_thoughts_logger.info(f"Plan step {i+1}: {step}")

                return parsed_steps
            else:
                logger.error("Invalid plan format returned from LLM")
                return None

        except Exception as e:
            logger.error(f"Error creating multi-step plan: {e}", exc_info=True)
            return None

    async def verify_step_completion(self, step_description: str, current_page_details: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies if a step has been completed successfully by analyzing the page state."""
        if not self.gemini_model:
            return {"verified": False, "confidence": 0, "message": "LLM not available"}

        try:
            html_content = current_page_details.get("html_content", "")
            url = current_page_details.get("url", "unknown")
            title = current_page_details.get("title", "unknown")

            # Truncate HTML if needed
            MAX_VERIFY_HTML_LENGTH = 8000
            prompt_html_content = html_content[:MAX_VERIFY_HTML_LENGTH] if len(
                html_content) > MAX_VERIFY_HTML_LENGTH else html_content

            prompt = f"""
            You are a web automation verification system. Determine if the following step has been successfully completed based on the current page state:
            
            STEP TO VERIFY: {step_description}
            
            CURRENT PAGE STATE:
            URL: {url}
            TITLE: {title}
            HTML CONTENT EXCERPT:
            ```html
            {prompt_html_content}
            ```
            
            INSTRUCTIONS:
            1. Analyze the page state to determine if the step appears to have been completed
            2. Look for indicators like: URL changes, element presence/absence, content changes, etc.
            3. Provide your confidence level (0-100) and explanation
            
            Return ONLY a JSON object with the following structure:
            ```json
            {{
              "verified": true|false,
              "confidence": 0-100,
              "explanation": "Your explanation of why you believe the step was completed or not"
            }}
            ```
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            verification_result = json.loads(text.strip())
            logger.info(
                f"Step verification for '{step_description}': {verification_result}")
            return verification_result

        except Exception as e:
            logger.error(
                f"Error verifying step completion: {e}", exc_info=True)
            return {"verified": False, "confidence": 0, "message": f"Verification error: {str(e)}"}

    async def adapt_plan_if_needed(self, current_plan: List[str], executed_step: str, verification_result: Dict[str, Any], current_page_details: Dict[str, Any]) -> Optional[List[str]]:
        """Adapts the multi-step plan if verification shows the execution didn't go as expected."""
        if not verification_result.get("verified", False) and verification_result.get("confidence", 0) > 30:
            logger.info("Step verification failed - adapting plan")

            # Create a new plan based on the current state
            return await self.create_multi_step_plan(current_page_details)

        # If verification succeeded or we're not confident it failed, keep the current plan
        return current_plan

    async def start_execution(self, goal: str, websocket) -> bool:
        if self.is_running:
            logger.warning("Execution already in progress.")
            return False

        self.goal = goal
        self.completed_steps = []
        self.is_running = True
        self.is_paused = False
        self.is_waiting_user = False
        self.websocket = websocket
        self.last_step_description = None

        success = await self.browser.initialize()
        if not success:
            await self._send_message("error", {"message": "Failed to initialize browser"})
            self.is_running = False
            return False

        # Use advanced execution with look-ahead planning
        self.task = asyncio.create_task(self._advanced_execution_loop())
        logger.info(f"Starting advanced execution towards goal: {self.goal}")
        await self._send_message("execution_started", {"goal": self.goal})
        return True

    async def _advanced_execution_loop(self):
        """Enhanced execution loop that uses multi-step planning, verification and error recovery"""
        logger.info(
            "Starting advanced execution loop with look-ahead planning and error recovery")
        last_page_details_analyzed = None
        step_counter = 0
        current_plan = None
        current_plan_index = 0

        try:
            # Initial page loading
            page_details = await self.browser.get_page_details()
            if not page_details.get("success", False):
                await self._handle_failure(f"Failed to get initial page details: {page_details.get('message')}", None, page_details)
                return

            # Initial multi-step planning
            await self._send_message("planning_step", {"message": "Creating execution plan..."})
            current_plan = await self.create_multi_step_plan(page_details)
            if not current_plan or not isinstance(current_plan, list) or len(current_plan) == 0:
                await self._handle_failure("Failed to create initial plan.", None, page_details)
                return

            # Send the plan to the UI
            await self._send_message("plan_created", {"plan": current_plan})
            logger.info(f"Created plan with {len(current_plan)} steps")

            while self.is_running and current_plan_index < len(current_plan):
                if self.is_paused:
                    await asyncio.sleep(0.5)
                    continue

                # Check for error states and attempt recovery before proceeding
                error_state = await self.browser.detect_error_state()
                if error_state.get("hasError", False):
                    error_type = error_state.get("errorType")
                    error_message = error_state.get(
                        "errorMessage", "Unknown error")

                    logger.warning(
                        f"Detected error state: {error_type} - {error_message}")
                    await self._send_message("agent_thought", {
                        "thought": f"Detected potential issue: {error_message}. Attempting automatic recovery..."
                    })

                    # Try to recover automatically
                    recovery_result = await self.browser.attempt_recovery(error_state)
                    if recovery_result.get("success", True):
                        action_taken = recovery_result.get(
                            "action_taken", "none")
                        if action_taken != "none":
                            logger.info(
                                f"Automatic recovery succeeded with action: {action_taken}")
                            await self._send_message("agent_thought", {
                                "thought": f"Automatic recovery succeeded: {recovery_result.get('message')}"
                            })

                            # Brief pause to let the recovery take effect
                            await asyncio.sleep(1)
                            continue  # Skip to the next loop iteration to reassess the page
                    else:
                        # If recovery failed and it's a serious error that would prevent execution,
                        # we might need to involve the user
                        if error_type in ['captcha', 'authentication_required']:
                            await self._handle_failure(
                                f"Automatic recovery not possible: {error_message}. User intervention needed.",
                                {"error_type": error_type},
                                page_details
                            )
                            return

                # Get the current step from the plan
                current_step = current_plan[current_plan_index]
                self.last_step_description = current_step

                # Notify about executing the step
                await self._send_message("executing_step", {
                    "step_index": step_counter,
                    "step": current_step
                })

                # Get the latest page details
                page_details = await self.browser.get_page_details()
                if not page_details.get("success", False):
                    await self._handle_failure(f"Failed to get page details: {page_details.get('message')}", None, page_details)
                    continue

                # Use JavaScript-based analysis to determine the action
                await self._send_message("analyzing", {
                    "message": f"Analyzing page for step: {current_step}"
                })

                analysis = await self.analyze_possible_actions(page_details, current_step)
                if not analysis.get("success", False):
                    # Fall back to HTML-based analysis
                    logger.warning(
                        "JavaScript analysis failed, falling back to HTML analysis")
                    await self._send_message("analyzing", {
                        "message": f"JavaScript scan failed, using HTML analysis for step: {current_step}"
                    })
                    analysis = await self.analyze_state(page_details, current_step)
                    if not analysis.get("success", False):
                        await self._handle_failure(
                            f"Analysis failed: {analysis.get('message')}", None, page_details)
                        continue

                action = analysis.get("action", {})
                action_type = action.get("action_type")

                if action_type == "complete":
                    logger.info(
                        f"Step '{current_step}' deemed complete by analysis.")
                    self.completed_steps.append(
                        f"{current_step} (Auto-completed)")
                    await self._send_message("step_completed", {
                        "step_index": step_counter,
                        "message": f"Completed: {current_step} (Auto-completed)"
                    })

                    # Move to next step in plan
                    current_plan_index += 1
                    step_counter += 1
                    continue

                # Execute the action
                await self._send_message("action", {
                    "type": action_type,
                    "details": action
                })

                result = await self.execute_action(action)

                # Check for navigation or network errors immediately after action
                if not result.get("success", False):
                    # Before giving up, check if we hit a common error state that
                    # we can recover from automatically
                    error_state = await self.browser.detect_error_state()
                    if error_state.get("hasError", False):
                        recovery_result = await self.browser.attempt_recovery(error_state)
                        if recovery_result.get("success", False):
                            # If recovery worked, retry the action
                            # Give the page time to stabilize
                            await asyncio.sleep(1)
                            # Retry
                            result = await self.execute_action(action)

                    if not result.get("success", False):
                        # If we still failed after recovery attempt (or if no recovery was possible)
                        await self._handle_failure(
                            f"Action failed: {result.get('message')}", action, page_details)
                        continue

                # Update page details after the action
                updated_page_details = await self.browser.get_page_details()
                if not updated_page_details.get("success", False):
                    logger.warning(
                        "Could not get updated page details after action.")
                    updated_page_details = page_details  # Use previous details as fallback

                # Verify step completion
                verification = await self.verify_step_completion(current_step, updated_page_details)
                verified = verification.get("verified", False)
                confidence = verification.get("confidence", 0)
                explanation = verification.get(
                    "explanation", "No explanation provided.")

                agent_thoughts_logger.info(
                    f"Step verification: {explanation} (Verified: {verified}, Confidence: {confidence})")
                await self._send_message("agent_thought", {"thought": f"Verification: {explanation}"})

                if verified or confidence < 70:  # If verified or we're not confident it failed
                    # Add to completed steps
                    self.completed_steps.append(current_step)
                    await self._send_message("step_completed", {
                        "step_index": step_counter,
                        "message": f"Completed: {current_step}"
                    })

                    # Move to the next step in the plan
                    current_plan_index += 1
                    step_counter += 1

                    # Check if the overall goal is met
                    if current_plan_index >= len(current_plan):
                        goal_met = await self.check_goal_completion(updated_page_details)
                        if goal_met:
                            await self._send_message("execution_complete", {"message": "Overall goal achieved."})
                            break
                        else:
                            # If we've completed the plan but the goal isn't met, create a new plan
                            await self._send_message("planning_step", {"message": "Creating follow-up plan..."})
                            new_plan = await self.create_multi_step_plan(updated_page_details)
                            if new_plan and len(new_plan) > 0:
                                current_plan = new_plan
                                current_plan_index = 0
                                await self._send_message("plan_created", {"plan": current_plan})
                                logger.info(
                                    f"Created new plan with {len(current_plan)} steps")
                            else:
                                await self._handle_failure("Failed to create new plan after completing all steps.", None, updated_page_details)
                                break
                else:
                    # Step wasn't verified, adapt the plan
                    logger.info(
                        f"Step verification failed with confidence {confidence}. Adapting plan...")
                    await self._send_message("analyzing", {
                        "message": f"Step verification failed ({confidence}% confidence). Adapting plan..."
                    })

                    adapted_plan = await self.adapt_plan_if_needed(
                        current_plan[current_plan_index:],
                        current_step,
                        verification,
                        updated_page_details
                    )

                    if adapted_plan and len(adapted_plan) > 0:
                        # Replace remaining steps with adapted plan
                        new_full_plan = self.completed_steps + adapted_plan
                        current_plan = adapted_plan
                        current_plan_index = 0  # Reset to start of the new adapted plan

                        # Notify about plan changes
                        await self._send_message("plan_created", {"plan": new_full_plan})
                        logger.info(
                            f"Adapted plan with {len(adapted_plan)} new steps")
                    else:
                        # If we can't adapt the plan, continue with existing one
                        logger.warning(
                            "Could not adapt plan, continuing with current plan")
                        current_plan_index += 1
                        step_counter += 1

                # Brief pause between actions
                await asyncio.sleep(0.5)

            if not self.is_paused:
                logger.info("Execution loop finished")

        except asyncio.CancelledError:
            logger.info("Execution task cancelled.")
        except Exception as e:
            logger.error(
                f"Unexpected error in execution loop: {e}", exc_info=True)
            await self._handle_failure(f"Unexpected error: {str(e)}", None, last_page_details_analyzed)
        finally:
            self.is_running = False
            logger.info("Advanced execution loop stopped.")

    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action determined by the LLM."""
        if not action:
            return {"success": False, "message": "No action to execute"}

        action_type = action.get("action_type", "").lower()
        selector = action.get("selector", "")
        value = action.get("value", "")
        explanation = action.get("explanation", "")

        logger.info(
            f"Executing action: {action_type} with values: {selector=}, {value=}")

        result = {"success": False, "message": "Unknown action type"}

        # Before action execution, capture page state for verification
        before_state = {}
        try:
            screenshot_result = await self.browser.capture_screenshot()
            if screenshot_result.get("success"):
                before_state["screenshot"] = screenshot_result.get(
                    "screenshot")
        except Exception as e:
            logger.warning(f"Failed to capture before-screenshot: {e}")

        # Execute the appropriate action
        try:
            if action_type == "navigate":
                result = await self.browser.navigate(value)

                # Handle navigation errors and attempt recovery
                if not result.get("success", False):
                    error_msg = result.get("message", "")
                    logger.warning(f"Navigation failed: {error_msg}")

                    # Check if it's a protocol error (missing http://)
                    if "invalid URL" in error_msg or "Protocol error" in error_msg:
                        # This should be fixed by our browser.py update, but let's add a backup recovery
                        fixed_url = value
                        if not fixed_url.startswith(('http://', 'https://')):
                            fixed_url = f"https://{value}"
                            logger.info(
                                f"Retrying navigation with fixed URL: {fixed_url}")
                            result = await self.browser.navigate(fixed_url)

            elif action_type == "click":
                result = await self.browser.click(selector)

            elif action_type == "type":
                result = await self.browser.type_text(selector, value)

            elif action_type == "wait":
                try:
                    wait_time = float(value) if value else 1.0
                    # Limit between 0.1 and 10 seconds
                    wait_time = min(10.0, max(0.1, wait_time))
                    await asyncio.sleep(wait_time)
                    result = {"success": True,
                              "message": f"Waited for {wait_time} seconds"}
                except ValueError:
                    result = {"success": False,
                              "message": f"Invalid wait duration: {value}"}

            elif action_type == "javascript":
                result = await self.browser.execute_javascript(value)

            elif action_type == "complete":
                result = {"success": True,
                          "message": "Task marked as complete"}

            else:
                result = {"success": False,
                          "message": f"Unsupported action type: {action_type}"}
        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            result = {"success": False,
                      "message": f"Action '{action_type}' failed. Error: {e}"}

        # After successful action execution, verify the change
        if result.get("success"):
            try:
                # Check for error states like cookie dialogs, popups, etc.
                error_state = await self.browser.detect_error_state()

                if error_state.get("hasError", False):
                    error_type = error_state.get("errorType")
                    error_msg = error_state.get("errorMessage")
                    logger.warning(
                        f"Detected error state after action: {error_type}: {error_msg}")

                    # Attempt automatic recovery
                    recovery_result = await self.browser.attempt_recovery(error_state)

                    if recovery_result.get("success"):
                        logger.info(
                            f"Auto-recovery successful: {recovery_result.get('message')}")
                        # Update result to indicate recovery was needed but succeeded
                        result["recovery_needed"] = True
                        result["recovery_action"] = recovery_result.get(
                            "action_taken")
                        result["recovery_message"] = recovery_result.get(
                            "message")
                    else:
                        logger.warning(
                            f"Auto-recovery failed: {recovery_result.get('message')}")
                        # If recovery failed and this is a blocking error, report it
                        if error_type in ["captcha", "authentication_required"]:
                            result["success"] = False
                            result["message"] = f"Blocked by {error_type}: {error_msg}"

                # Capture a screenshot after action to visually verify state change
                after_screenshot_result = await self.browser.capture_screenshot()
                if after_screenshot_result.get("success") and before_state.get("screenshot"):
                    # Compare before/after visual states
                    visual_comparison = await self.browser.compare_visual_states(
                        before_state.get("screenshot"),
                        after_screenshot_result.get("screenshot")
                    )

                    # Update the result with visual verification data
                    if visual_comparison.get("success"):
                        result["visual_change"] = visual_comparison.get(
                            "visual_change")
                        result["visual_difference_score"] = visual_comparison.get(
                            "difference_score", 0)

                    # Share the screenshot with the user via websocket if available
                    if self.websocket:
                        try:
                            await self.websocket.send_json({
                                "type": "browser_screenshot",
                                "screenshot": after_screenshot_result.get("screenshot")
                            })
                        except Exception as ws_error:
                            logger.error(
                                f"Failed to send screenshot via WebSocket: {ws_error}")

            except Exception as verify_error:
                logger.error(f"Post-action verification error: {verify_error}")

        # Update the original action dict with the execution result
        action["execution_result"] = result

        logger.info(
            f"Action execution result: {result.get('success')}: {result.get('message')}")
        return result

    async def _handle_failure(self, error_message: str, failed_action: Optional[Dict], page_details_at_failure: Optional[Dict]):
        logger.error(f"Execution failed: {error_message}")
        html_content_at_failure = page_details_at_failure.get(
            'html_content') if page_details_at_failure else None
        if html_content_at_failure:
            log_html_excerpt = html_content_at_failure[:2000] + (
                "..." if len(html_content_at_failure) > 2000 else "")
            logger.error(f"--- HTML Content at Failure (First 2000 chars) ---")
            logger.error(log_html_excerpt)
            logger.error(f"--- End HTML Content ---")
        else:
            logger.error("HTML content at failure was not available.")

        self.is_paused = True
        self.is_waiting_user = True
        self.failed_action = failed_action
        self.last_error_message = error_message

        await self._send_message("execution_paused", {
            "message": error_message,
            "failed_step_description": self.last_step_description,
            "failed_action": failed_action
        })

    async def _send_message(self, msg_type: str, data: Any = None):
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
        logger.debug("Resume execution called.")
        if not self.is_running:
            logger.warning("Cannot resume, execution is not running.")
            return False
        if not self.is_paused:
            logger.warning("Cannot resume, execution is not paused.")
            return False

        logger.info("Resuming execution. Will replan next step.")
        self.is_paused = False
        self.is_waiting_user = False
        self.last_error_message = None
        self.failed_action = None
        await self._send_message("resuming", {
            "message": "Resuming execution..."
        })
        logger.debug("Resume state updated and message sent.")
        return True

    async def stop_execution(self):
        self.is_running = False
        self.is_paused = False
        await self._send_message("execution_stopped", {
            "message": "Execution stopped by user"
        })

        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        return True
