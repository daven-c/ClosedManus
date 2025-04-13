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
        self.conversation_history: List[Dict[str, str]] = []  # Added
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

    async def start_execution(self, goal: str, websocket) -> bool:
        """Starts the execution loop for the given goal."""
        if self.is_running:
            logger.warning("Execution already in progress.")
            await self._send_message("error", {"message": "Execution already in progress."})
            return False

        self.goal = goal
        self.completed_steps = []
        # Initialize conversation history
        self.conversation_history = [{"role": "user", "content": goal}]
        self.is_running = True
        self.is_paused = False
        self.is_waiting_user = False
        self.websocket = websocket
        self.last_step_description = None
        self.last_error_message = None
        self.failed_action = None

        # Ensure browser is initialized
        if not self.browser.page or self.browser.page.is_closed():
            logger.info("Browser not ready for execution, initializing...")
            success = await self.browser.initialize()
            if not success:
                await self._send_message("error", {"message": "Failed to initialize browser for execution."})
                self.is_running = False  # Ensure state is correct
                return False
        else:
            logger.info("Browser already initialized.")

        # Start the execution loop as a background task
        self.task = asyncio.create_task(self._basic_execution_loop())
        logger.info(f"Starting execution towards goal: {self.goal}")
        await self._send_message("execution_started", {"goal": self.goal})
        return True

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
            available_elements = scan_result.get(
                "scan_result", {}) if scan_result.get("success") else {}

            # If scanning failed, log it but continue with basic HTML analysis
            if not scan_result.get("success"):
                logger.warning(
                    f"Element scanning failed, continuing with basic HTML analysis: {scan_result.get('message')}")
                available_elements = {
                    "warning": "Element scanning failed, relying on basic HTML analysis"}

            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])

            prompt = f"""
            You are a web automation expert. Your task is to analyze the current page state and determine the exact action needed for the current step. Prioritize asking the user if any information is unclear or missing.

            OVERALL GOAL: {self.goal}

            CONVERSATION HISTORY:
            {formatted_history}

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

            IMPORTANT:
            1. Check AVAILABLE ELEMENTS before suggesting actions. Use these elements whenever possible.
            2. If the step requires *any* information not immediately clear from the page or history (e.g., login credentials, specific choices, confirmation of intent), set action_type to "ask_user" and provide a clear, specific question in the "value" field. Err on the side of asking if unsure.
            3. **If the step involves searching (e.g., "Search for X", "Enter X into search bar"), determine if the query needs to be typed *or* if the search needs to be submitted.**
                - If the query needs typing, use the "type" action.
                - **If the query has just been typed (check conversation history/previous steps) and the current step is to perform the search or click the search button, use the "click" action on the appropriate search button (e.g., 'Google Search', 'Search').** Look for buttons near the search input in AVAILABLE ELEMENTS.
            4. Otherwise, determine the next browser action (navigate, click, type, wait, javascript, complete).

            Return a JSON object with the following structure:
            {{
                "action_type": "navigate|click|type|wait|javascript|complete|ask_user",
                "selector": "CSS selector or Playwright locator (null if ask_user or javascript)",
                "locator_strategy": "css|xpath|text|role|get_by_role|get_by_label (null if ask_user or javascript)",
                "locator_args": {{}}, # e.g., {{"role": "button", "name": "Google Search"}} for get_by_role
                "value": "text to type, URL, wait duration, JS code, or the question to ask the user",
                "javascript_code": "optional JavaScript code to execute (use instead of selector/strategy if needed)",
                "explanation": "explanation of why this action/question was chosen",
                "thought": "your reasoning process, including checking if search needs typing vs submitting"
            }}
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                }
            )

            text = response.text.strip()
            # Send the raw LLM response as the thought
            agent_thoughts_logger.info(
                f"Analysis for '{step}' (Raw LLM Response): {text}")
            await self._send_message("agent_thought", {"thought": f"Analysis for '{step}' (Raw LLM Response):\n```json\n{text}\n```"})

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            try:
                action = json.loads(text)

                if not isinstance(action, dict):
                    logger.error(
                        f"LLM response parsed into a list, not a dict: {action}")
                    if isinstance(action, list) and len(action) == 1 and isinstance(action[0], dict):
                        logger.warning(
                            "Assuming the first element of the list is the intended action.")
                        action = action[0]
                    else:
                        return {"success": False, "message": "LLM response did not parse into the expected dictionary format."}

                # Log the extracted thought if available, but the raw response was already sent
                thought = action.get(
                    "thought", "No thought provided in parsed JSON.")
                agent_thoughts_logger.info(
                    f"Parsed thought from analysis for '{step}': {thought}")

                if action.get("action_type") in ["click", "type"]:
                    if action.get("locator_strategy") == "get_by_role":
                        role_args = action.get("locator_args", {})
                        action["selector"] = f"role={role_args.get('role', '')}"
                    elif not action.get("selector"):
                        if not action.get("javascript_code"):
                            logger.warning(
                                f"Missing selector and javascript_code for action: {action}")

                return {"success": True, "action": action}

            except json.JSONDecodeError as e:
                # Log the text that failed parsing
                logger.error(f"Failed to parse JSON response: {text}")
                return {"success": False, "message": f"Failed to parse LLM response: {str(e)}"}

        except Exception as e:
            logger.error(f"Error analyzing state: {e}", exc_info=True)
            return {"success": False, "message": f"Analysis error: {str(e)}"}

    async def extract_final_answer(self, goal: str, final_page_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts the final answer from the page based on the original goal."""
        if not self.gemini_model:
            return {"success": False, "message": "LLM not initialized"}

        try:
            html_content = final_page_details.get("html_content", "")
            url = final_page_details.get("url", "unknown")
            title = final_page_details.get("title", "unknown")

            MAX_ANSWER_HTML_LENGTH = 15000
            prompt_html_content = html_content[:MAX_ANSWER_HTML_LENGTH] if len(
                html_content) > MAX_ANSWER_HTML_LENGTH else html_content

            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])

            prompt = f"""
            You are a web data extraction assistant. Based on the original goal, conversation history, and the final state of the web page, extract the specific answer requested.

            ORIGINAL GOAL: {goal}

            CONVERSATION HISTORY:
            {formatted_history}

            FINAL PAGE STATE:
            URL: {url}
            TITLE: {title}
            HTML CONTENT EXCERPT:
            ```html
            {prompt_html_content}
            ```

            INSTRUCTIONS:
            1. Carefully read the ORIGINAL GOAL and CONVERSATION HISTORY to understand what information needs to be extracted.
            2. Analyze the HTML CONTENT EXCERPT and the page URL/TITLE to find the information relevant to the goal/conversation.
            3. Extract the specific piece of information requested.
            4. If the information cannot be found, state that clearly in the 'answer' field and set 'success' to false.
            5. Provide your reasoning in the 'thought' field.

            Return ONLY a JSON object with the following structure:
            ```json
            {{
              "success": true|false,
              "answer": "The extracted answer string, or null/a descriptive message if not found",
              "thought": "Your reasoning for finding (or not finding) the answer on the page."
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

            text = response.text.strip()
            # Send the raw LLM response as the thought
            agent_thoughts_logger.info(
                f"Final Answer Extraction (Raw LLM Response): {text}")
            await self._send_message("agent_thought", {"thought": f"Final Answer Extraction (Raw LLM Response):\n```json\n{text}\n```"})

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            # Log the extracted thought if available, but the raw response was already sent
            thought = result.get(
                "thought", "No thought provided in parsed JSON.")
            agent_thoughts_logger.info(
                f"Parsed thought from Final Answer Extraction: {thought}")

            logger.info(f"LLM final answer extraction result: {result}")

            # Additional detail for better error handling
            if not result.get("success"):
                logger.warning(
                    "Answer extraction was not successful according to LLM.")
                if not result.get("answer"):
                    result["answer"] = "Could not extract the required information from the final page state."

            return result

        except Exception as e:
            logger.error(f"Error extracting final answer: {e}", exc_info=True)
            return {"success": False, "answer": "Error extracting answer from page", "message": f"Answer extraction error: {str(e)}"}

    async def _basic_execution_loop(self):
        """Basic execution loop that generates and executes one step at a time."""
        logger.info("Starting basic execution loop.")
        step_counter = 0
        final_page_details = None

        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(1.5)
                    continue
                if self.is_waiting_user:
                    await asyncio.sleep(1.0)
                    continue

                current_page_details = await self.browser.get_page_details()
                if not current_page_details.get("success", False):
                    await self._handle_failure(f"Failed to get page details: {current_page_details.get('message')}", None, current_page_details)
                    return
                final_page_details = current_page_details

                goal_met = await self.check_goal_completion(current_page_details)
                if goal_met:
                    logger.info("Goal completion check returned true.")
                    await self._send_message("status_update", {"message": "Goal condition met. Extracting final answer..."})
                    break

                next_step = await self.plan_next_step(current_page_details)
                if not next_step:
                    await self._handle_failure("Failed to determine the next step.", None, current_page_details)
                    return
                self.last_step_description = next_step

                await self._send_message("executing_step", {
                    "step_index": step_counter,
                    "step": next_step
                })

                analysis = await self.analyze_state(current_page_details, next_step)
                if not analysis.get("success", False):
                    await self._handle_failure(f"Analysis failed: {analysis.get('message')}", None, current_page_details)
                    return

                action = analysis.get("action", {})
                action_type = action.get("action_type")

                if action_type == "ask_user":
                    question = action.get(
                        "value", "I need more information to proceed. Can you please provide details?")
                    logger.info(f"Agent needs input: {question}")
                    self.conversation_history.append(
                        {"role": "agent", "content": question})
                    await self._send_message("request_user_input", {"message": question})
                    self.is_waiting_user = True
                    continue

                if action_type == "complete":
                    logger.info(
                        f"LLM analysis indicated completion after step '{next_step}'.")
                    self.completed_steps.append(
                        f"{next_step} (Marked as complete by LLM)")
                    await self._send_message("step_completed", {
                        "step_index": step_counter,
                        "message": f"Completed: {next_step} (LLM determined completion)"
                    })
                    await self._send_message("status_update", {"message": "LLM indicated task completion. Extracting final answer..."})
                    final_page_details = await self.browser.get_page_details()
                    break

                result = await self.execute_action(action)
                if not result.get("success", False):
                    final_page_details_on_fail = await self.browser.get_page_details()
                    await self._handle_failure(f"Action failed: {result.get('message')}", action, final_page_details_on_fail)
                    return

                self.completed_steps.append(next_step)
                await self._send_message("step_completed", {
                    "step_index": step_counter,
                    "message": f"Completed: {next_step}"
                })

                step_counter += 1
                await asyncio.sleep(0.5)

            if self.is_running and final_page_details:
                logger.info(
                    "Execution loop finished successfully. Attempting to extract final answer.")
                answer_result = await self.extract_final_answer(self.goal, final_page_details)

                final_message = "No specific answer extracted."
                if answer_result.get("success") and answer_result.get("answer"):
                    final_message = f"Extracted answer: {answer_result.get('answer')}"
                elif not answer_result.get("success") and answer_result.get("answer"):
                    final_message = f"Answer extraction failed: {answer_result.get('answer')}"

                await self._send_message("execution_complete", {
                    "message": "Task completed. " + final_message,
                    "final_answer_details": answer_result
                })

                if answer_result.get("answer"):
                    logger.info(
                        f"FINAL ANSWER/RESULT: {answer_result.get('answer')}")
                else:
                    logger.warning("No final answer was extracted.")

            elif self.is_running:
                logger.warning(
                    "Execution loop finished but final page details were not available.")
                await self._send_message("execution_complete", {
                    "message": "Goal achieved or process completed, but could not get final page details for answer extraction.",
                    "final_answer_details": {"success": False, "answer": None, "message": "Final page details unavailable."}
                })

            logger.info("Basic execution loop finished.")

        except asyncio.CancelledError:
            logger.info("Execution task cancelled.")
            await self._send_message("execution_stopped", {"message": "Execution cancelled."})
        except Exception as e:
            logger.error(
                f"Unexpected error in execution loop: {e}", exc_info=True)
            final_page_details_on_error = await self.browser.get_page_details() if self.browser.page else None
            await self._handle_failure(f"Unexpected error: {str(e)}", None, final_page_details_on_error)
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

        if action_type in ["click", "type"]:
            element_present = await self.browser.element_exists(selector)
            if not element_present:
                return {
                    "success": False,
                    "message": f"Element with selector '{selector}' not found on page"
                }

        logger.info(
            f"Executing action: {action_type} with values: {selector=}, {value=}")

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
        self.is_running = False

        await self._send_message("execution_failed", {
            "message": error_message,
            "failed_step_description": self.last_step_description,
            "failed_action": failed_action
        })
        logger.info("Execution marked as failed and stopped.")

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
        if self.last_error_message:
            logger.warning(
                "Cannot resume after a failure. Please start a new execution.")
            await self._send_message("error", {"message": "Cannot resume after failure. Please start again."})
            return False

        if self.is_waiting_user:
            logger.warning("Cannot resume, agent is waiting for user input.")
            await self._send_message("error", {"message": "Agent is waiting for your input."})
            return False

        if not self.is_running:
            logger.warning("Cannot resume, execution is not running.")
            return False
        if not self.is_paused:
            logger.warning("Cannot resume, execution is not paused.")
            return False

        logger.info("Resuming execution.")
        self.is_paused = False
        self.is_waiting_user = False
        await self._send_message("resuming", {
            "message": "Resuming execution..."
        })
        logger.debug("Resume state updated and message sent.")
        return True

    async def handle_user_input(self, user_message: str):
        """Handles input provided by the user when the agent is waiting."""
        logger.debug(f"handle_user_input called with: {user_message}")
        # Check both is_running and is_waiting_user before proceeding
        if self.is_running and self.is_waiting_user:
            logger.info(f"Received user input while waiting: {user_message}")
            self.conversation_history.append(
                {"role": "user", "content": user_message})
            self.is_waiting_user = False  # Set the flag to allow the loop to continue
            # Add specific log
            logger.info(
                "Set is_waiting_user to False. Agent loop should resume on next iteration.")
            await self._send_message("user_input_received", {"message": "Input received, resuming..."})
            # The loop will automatically continue on the next iteration
        elif not self.is_running:
            logger.warning(
                "Received user input, but execution is not running.")
            await self._send_message("error", {"message": "Received input, but execution is not active."})
            # No return needed here as the function ends
        elif not self.is_waiting_user:
            logger.warning(
                "Received user input, but the agent was not waiting for it.")
            # Optionally inform the user, or just log it.
            # await self._send_message("info", {"message": "Received input, but I wasn't waiting for any."})
            # No return needed here

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

    async def plan_next_step(self, current_page_details: Dict[str, Any]) -> Optional[str]:
        """Determines the single next high-level step based on the goal and current state."""
        if not self.gemini_model:
            logger.warning("LLM not available, cannot plan next step.")
            return None

        try:
            html_content = current_page_details.get("html_content", "")
            url = current_page_details.get("url", "unknown")
            title = current_page_details.get("title", "unknown")

            MAX_PLAN_HTML_LENGTH = 4000  # Keep relatively small for planning
            prompt_html_content = html_content[:MAX_PLAN_HTML_LENGTH] + (
                "..." if len(html_content) > MAX_PLAN_HTML_LENGTH else "")

            history = "\n".join(f"- {s}" for s in self.completed_steps)
            if not history:
                history = "No steps completed yet."

            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])

            prompt = f"""
            You are a web automation planner determining the *single next step*. Prioritize asking for user input if clarification is needed. Break down complex tasks like searching into multiple steps.

            OVERALL GOAL: {self.goal}

            CONVERSATION HISTORY:
            {formatted_history}

            COMPLETED STEPS HISTORY:
            {history}

            CURRENT WEB PAGE STATE:
            URL: {url}
            TITLE: {title}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            Based on the goal, conversation, history, and current state, determine the **single most logical next step**.
            - If the next logical action requires *any* clarification or input from the user (e.g., missing credentials, ambiguous choices, confirmation of the next action), the next step should be to ask the user (e.g., "Ask user for login credentials", "Ask user to confirm proceeding with checkout").
            - **If the task involves searching:**
                - If the search query hasn't been typed yet, the next step is "Type '[search query]' into the search bar".
                - **If the search query *was* typed in the previous step (check COMPLETED STEPS HISTORY), the next step is "Click the search button" or "Submit the search".**
            - Otherwise, describe the high-level browser action (e.g., "Click the 'Login' button", "Fill in the search bar").
            - Consider the history to avoid repetition and ensure progress.

            **Output Format (JSON only):**
            Return ONLY a JSON object containing the next step description and your reasoning.
            ```json
            {{
                "next_step": "Description of the single next step (e.g., 'Type '10 richest people' into search bar', 'Click the Google Search button', 'Ask user for the specific product code').",
                "thought": "Your reasoning for choosing this step, considering if a search query was just typed."
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
            text = response.text.strip()
            # Send the raw LLM response as the thought
            agent_thoughts_logger.info(
                f"Planning next step (Raw LLM Response): {text}")
            await self._send_message("agent_thought", {"thought": f"Planning next step (Raw LLM Response):\n```json\n{text}\n```"})

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            next_step = result.get("next_step")
            # Log the extracted thought if available, but the raw response was already sent
            thought = result.get(
                "thought", "No thought provided in parsed JSON.")
            agent_thoughts_logger.info(
                f"Parsed thought from Planning next step: {thought}")

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
        """Checks if the overall goal has been met based on the current state."""
        if not self.gemini_model:
            logger.warning("LLM not available, assuming goal not met.")
            return False

        try:
            html_content = current_page_details.get("html_content", "")
            url = current_page_details.get("url", "unknown")
            title = current_page_details.get("title", "unknown")

            MAX_CHECK_HTML_LENGTH = 4000  # Keep relatively small for checking
            prompt_html_content = html_content[:MAX_CHECK_HTML_LENGTH] + (
                "..." if len(html_content) > MAX_CHECK_HTML_LENGTH else "")

            history = "\n".join(f"- {s}" for s in self.completed_steps)

            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])

            prompt = f"""
            You are a web automation goal checker.

            OVERALL GOAL: {self.goal}

            CONVERSATION HISTORY:
            {formatted_history}

            COMPLETED STEPS HISTORY:
            {history}

            CURRENT WEB PAGE STATE:
            URL: {url}
            TITLE: {title}
            HTML CONTENT EXCERPT (potentially truncated):
            ```html
            {prompt_html_content}
            ```

            **Instructions:**
            Based *only* on the OVERALL GOAL, CONVERSATION HISTORY, COMPLETED STEPS HISTORY, and the CURRENT WEB PAGE STATE, determine if the overall goal has been successfully achieved.
            - Consider if the current state or conversation indicates success.

            **Output Format (JSON only):**
            Return ONLY a JSON object with a boolean value and your reasoning.
            ```json
            {{
                "goal_achieved": true | false,
                "thought": "Your reasoning."
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
            text = response.text.strip()
            # Send the raw LLM response as the thought
            agent_thoughts_logger.info(
                f"Goal check (Raw LLM Response): {text}")
            await self._send_message("agent_thought", {"thought": f"Goal check (Raw LLM Response):\n```json\n{text}\n```"})

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            goal_achieved = result.get("goal_achieved", False)
            # Log the extracted thought if available, but the raw response was already sent
            thought = result.get(
                "thought", "No thought provided in parsed JSON.")
            agent_thoughts_logger.info(
                f"Parsed thought from Goal check: {thought}")

            logger.info(f"LLM goal check result: {goal_achieved}")
            return goal_achieved
        except Exception as e:
            logger.error(f"Error checking goal completion: {e}", exc_info=True)
            return False
