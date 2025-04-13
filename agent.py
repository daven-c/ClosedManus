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
        """Analyzes the current page state and determines the next action to take."""
        if not self.gemini_model:
            return {"success": False, "message": "LLM not initialized"}

        try:
            html_content = page_details.get('html_content', '')
            url = page_details.get('url', 'unknown')
            title = page_details.get('title', 'unknown')

            # First, scan for available elements
            scan_result = await self.browser.scan_actionable_elements()
            available_elements = scan_result.get("scan_result", {}) if scan_result.get("success") else {}

            prompt = f"""
            You are a web automation expert. Analyze the current page state and determine the exact action needed for this step.
            
            STEP TO EXECUTE: {step}
            
            CURRENT PAGE STATE:
            URL: {url}
            TITLE: {title}

            AVAILABLE ELEMENTS ON PAGE:
            {json.dumps(available_elements, indent=2)}

            HTML CONTENT:
            ```html
            {html_content[:18000]}
            ```
            
            IMPORTANT: Before suggesting any action:
            1. Check if the required element exists in the AVAILABLE ELEMENTS list
            2. Only suggest clicking or typing in elements that are confirmed present
            3. If an element is not found, suggest a 'wait' action or alternative approach

            Return a JSON object with the following structure:
            {{
                "action_type": "navigate|click|type|wait|javascript|complete",
                "selector": "CSS selector or Playwright locator",
                "locator_strategy": "css|xpath|text|role|get_by_role|get_by_label",
                "locator_args": {{}},
                "value": "text to type or URL to navigate to",
                "javascript_code": "optional JavaScript code to execute",
                "explanation": "explanation of why this action was chosen",
                "thought": "your reasoning process"
            }}
            """

            # Rest of the existing analyze_state code...

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                }
            )

            # Extract the response text and clean it up
            text = response.text.strip()
            print(type(text), text)
            
            # Remove any markdown code block indicators and get just the JSON content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
                text = text[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                text = text[0]
            
            print(text)
            
            # Parse the JSON
            try:
                action = json.loads(text)

                # Validate and format the action
                if action["action_type"] in ["click", "type"]:
                    if action.get("locator_strategy") == "get_by_role":
                        role_args = action.get("locator_args", {})
                        action["selector"] = f"role={role_args.get('role', '')}"
                    elif not action.get("selector"):
                        return {"success": False, "message": "Missing selector for click/type action"}

                return {"success": True, "action": action}

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {text}")
                return {"success": False, "message": f"Failed to parse LLM response: {str(e)}"}

        except Exception as e:
            logger.error(f"Error analyzing state: {e}", exc_info=True)
            return {"success": False, "message": f"Analysis error: {str(e)}"}

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
        self.task = asyncio.create_task(self._basic_execution_loop())
        logger.info(f"Starting advanced execution towards goal: {self.goal}")
        await self._send_message("execution_started", {"goal": self.goal})
        return True

    async def _basic_execution_loop(self):
        """Basic execution loop that generates and executes one step at a time."""
        logger.info("Starting basic execution loop.")
        step_counter = 0

        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(1.5)
                    continue

                # Get the current page details
                page_details = await self.browser.get_page_details()
                if not page_details.get("success", False):
                    await self._handle_failure(f"Failed to get page details: {page_details.get('message')}", None, page_details)
                    return

                # Check if the overall goal is met
                goal_met = await self.check_goal_completion(page_details)
                if goal_met:
                    await self._send_message("execution_complete", {"message": "Overall goal achieved."})
                    break

                # Generate the next step based on the current HTML
                await self._send_message("planning_step", {"message": "Determining the next step..."})
                next_step = await self.plan_next_step(page_details)
                if not next_step:
                    await self._handle_failure("Failed to determine the next step.", None, page_details)
                    return

                # Notify about the step being executed
                await self._send_message("executing_step", {
                    "step_index": step_counter,
                    "step": next_step
                })

                # Analyze the page and determine the action for the step
                analysis = await self.analyze_state(page_details, next_step)
                if not analysis.get("success", False):
                    await self._handle_failure(f"Analysis failed: {analysis.get('message')}", None, page_details)
                    return

                action = analysis.get("action", {})
                action_type = action.get("action_type")

                if action_type == "complete":
                    logger.info(f"Step '{next_step}' deemed complete by analysis.")
                    self.completed_steps.append(f"{next_step} (Auto-completed)")
                    await self._send_message("step_completed", {
                        "step_index": step_counter,
                        "message": f"Completed: {next_step} (Auto-completed)"
                    })
                    step_counter += 1
                    continue

                # Execute the action
                result = await self.execute_action(action)
                if not result.get("success", False):
                    await self._handle_failure(f"Action failed: {result.get('message')}", action, page_details)
                    return

                # Mark the step as completed
                self.completed_steps.append(next_step)
                await self._send_message("step_completed", {
                    "step_index": step_counter,
                    "message": f"Completed: {next_step}"
                })

                step_counter += 1
                await asyncio.sleep(0.5)

            logger.info("Basic execution loop finished.")

        except asyncio.CancelledError:
            logger.info("Execution task cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in execution loop: {e}", exc_info=True)
            await self._handle_failure(f"Unexpected error: {str(e)}", None, None)
        finally:
            self.is_running = False
            logger.info("Basic execution loop stopped.")

    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action determined by the LLM."""
        if not action:
            return {"success": False, "message": "No action to execute"}

        action_type = action.get("action_type", "").lower()
        selector = action.get("selector", "")
        value = action.get("value", "")

        # Verify element presence before interaction
        if action_type in ["click", "type"]:
            element_present = await self.browser.element_exists(selector)
            if not element_present:
                return {
                    "success": False, 
                    "message": f"Element with selector '{selector}' not found on page"
                }
        #explanation = action.get("explanation", "")

        logger.info(
            f"Executing action: {action_type} with values: {selector=}, {value=}")

        # Validate the selector for actions that require it
        if action_type in ["click", "type"] and not selector:
            logger.error(f"Invalid or missing selector for action: {action}")
            return {"success": False, "message": "Invalid or missing selector for action"}

        result = {"success": False, "message": "Unknown action type"}

        try:
            if action_type == "navigate":
                result = await self.browser.navigate(value)

            elif action_type == "click":
                result = await self.browser.click(selector)

            elif action_type == "type":
                result = await self.browser.type_text(selector, value)

            elif action_type == "wait":
                try:
                    wait_time = float(value) if value else 1.0
                    wait_time = min(10.0, max(0.1, wait_time))  # Limit between 0.1 and 10 seconds
                    await asyncio.sleep(wait_time)
                    result = {"success": True, "message": f"Waited for {wait_time} seconds"}
                except ValueError:
                    result = {"success": False, "message": f"Invalid wait duration: {value}"}

            elif action_type == "javascript":
                result = await self.browser.execute_javascript(value)

            elif action_type == "complete":
                result = {"success": True, "message": "Task marked as complete"}

            else:
                result = {"success": False, "message": f"Unsupported action type: {action_type}"}

        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            result = {"success": False, "message": f"Action '{action_type}' failed. Error: {e}"}

        logger.info(f"Action execution result: {result.get('success')}: {result.get('message')}")
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
