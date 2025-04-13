import logging
import json
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import google.generativeai as genai

# Set up logging
logger = logging.getLogger("web_automation")


class TelemetryManager:
    """Enterprise telemetry manager to track runtime metrics and LLM usage"""

    def __init__(self):
        self.session_start_time = time.time()
        self.metrics = {
            "llm_calls": 0,
            "llm_tokens_used": 0,
            "actions_executed": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "avg_response_time_ms": 0,
            "response_times": []
        }
        self.action_history = []
        self.error_history = []

    def record_llm_call(self, prompt_size: int, response_size: int, response_time_ms: float):
        """Record metrics from an LLM API call"""
        self.metrics["llm_calls"] += 1
        # Estimate token count roughly (could be more precise with a tokenizer)
        estimated_tokens = (prompt_size + response_size) / 4
        self.metrics["llm_tokens_used"] += estimated_tokens

        # Update response time metrics
        self.metrics["response_times"].append(response_time_ms)
        self.metrics["avg_response_time_ms"] = sum(
            self.metrics["response_times"]) / len(self.metrics["response_times"])

    def record_action(self, action_type: str, description: str, success: bool, duration_ms: float):
        """Record details about an executed action"""
        self.metrics["actions_executed"] += 1
        if success:
            self.metrics["successful_actions"] += 1
        else:
            self.metrics["failed_actions"] += 1

        self.action_history.append({
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "description": description,
            "success": success,
            "duration_ms": duration_ms
        })

    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Record detailed error information"""
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        })

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get current session metrics"""
        session_duration = time.time() - self.session_start_time

        return {
            "session_duration_seconds": session_duration,
            "metrics": self.metrics,
            "success_rate": self.metrics["successful_actions"] / max(1, self.metrics["actions_executed"]),
            "errors_count": len(self.error_history),
            "recent_errors": self.error_history[-5:] if self.error_history else []
        }


class LLMService:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.telemetry = TelemetryManager()

    async def create_plan(self, high_level_goal: str, state: Dict[str, Any]) -> Optional[List[str]]:
        """Generate a high-level plan to achieve the given goal based on the current state."""
        if not self.gemini_model:
            logger.warning("LLM not available, cannot create plan.")
            return None

        try:
            start_time = time.time()

            prompt = f"""
            You are a web automation assistant. Based on the current page state, create a concise step-by-step plan (2-5 steps) to achieve the following high-level goal: {high_level_goal}.
            CURRENT WEB PAGE STATE:
            URL: {state.get("url", "unknown")}
            TITLE: {state.get("title", "unknown")}
            CONTENT EXCERPT (first 1500 chars): {state.get("content", "")[:1500]}

            Focus on actionable steps from the current state.
            Return ONLY a JSON array of strings representing the steps.
            """
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "response_mime_type": "application/json"
                }
            )

            response_time_ms = (time.time() - start_time) * 1000
            self.telemetry.record_llm_call(
                prompt_size=len(prompt),
                response_size=len(response.text),
                response_time_ms=response_time_ms
            )

            steps = self._parse_json_response(response.text)
            if isinstance(steps, list) and all(isinstance(step, str) for step in steps):
                logger.info(f"Generated plan: {steps}")
                return steps
            else:
                logger.error(
                    f"LLM plan generation failed to return a valid list: {steps}")
                self.telemetry.record_error(
                    "plan_generation_failure",
                    "Failed to generate a valid plan",
                    {"response": response.text}
                )
                return None
        except Exception as e:
            logger.error(f"Error during plan generation: {e}")
            self.telemetry.record_error(
                "plan_generation_exception",
                str(e),
                {"goal": high_level_goal}
            )
            return None

    async def analyze_state_and_get_action(self, state: Dict[str, Any], step: str) -> Dict[str, Any]:
        """Analyze current browser state and determine the next specific action using the LLM."""
        if not self.gemini_model:
            logger.warning("LLM not available, cannot analyze state.")
            return {"success": False, "message": "LLM not initialized"}

        try:
            start_time = time.time()

            prompt = f"""
            You are a web automation assistant. Analyze the current page state and the current plan step to determine the single, most appropriate browser action to perform next. Focus on finding elements that are VISIBLE and INTERACTABLE.

            CURRENT PLAN STEP: {step}

            CURRENT WEB PAGE STATE:
            URL: {state.get('url')}
            TITLE: {state.get('title')}
            CONTENT (first 1500 chars): {state.get('content', '')[:1500]}

            Choose ONE action from: navigate, click, type, wait, complete.
            - 'navigate': Provide the URL in 'value'.
            - 'click': Provide a robust CSS selector for a VISIBLE and CLICKABLE element in 'selector'. Prefer selectors using text content (e.g., button:has-text("Submit")) or unique IDs/attributes if available. Avoid selectors prone to breaking (e.g., complex paths, index-based). If using aria-label, ensure it accurately reflects a visible element.
            - 'type': Provide a robust CSS selector for a VISIBLE input field in 'selector' and the text in 'value'.
            - 'wait': Provide the duration in seconds (e.g., 1 or 2) in 'value'. Use ONLY if waiting for a specific element to appear is not feasible via Playwright's waits.
            - 'complete': Use if the current step's goal is clearly achieved based on the page state (e.g., expected text is present, URL has changed appropriately).

            Return ONLY a JSON object with the following structure:
            {{
                "action_type": "navigate|click|type|wait|complete",
                "selector": "CSS selector if applicable, otherwise null",
                "value": "URL, text to type, or wait duration if applicable, otherwise null",
                "explanation": "Brief justification for choosing this action and selector/value based on the step and VISIBLE page elements."
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

            response_time_ms = (time.time() - start_time) * 1000
            self.telemetry.record_llm_call(
                prompt_size=len(prompt),
                response_size=len(response.text),
                response_time_ms=response_time_ms
            )

            action = self._parse_json_response(response.text)
            if isinstance(action, dict) and action.get("action_type"):
                logger.info(f"Determined action: {action}")
                return {"success": True, "action": action}
            else:
                logger.error(
                    f"LLM action generation failed to return a valid action object: {action}")
                self.telemetry.record_error(
                    "action_generation_failure",
                    "Failed to determine a valid action",
                    {"step": step, "response": response.text}
                )
                return {"success": False, "message": "Failed to determine a valid action from LLM response."}
        except Exception as e:
            logger.error(f"Error during state analysis: {e}")
            self.telemetry.record_error(
                "state_analysis_exception",
                str(e),
                {"step": step}
            )
            return {"success": False, "message": f"State analysis failed: {e}"}

    async def recalculate_remaining_plan(
        self,
        original_plan: List[str],
        current_step_index: int,
        state: Dict[str, Any],
        last_error: Optional[str] = None,
        failed_action: Optional[Dict] = None
    ) -> Optional[List[str]]:
        """Generate a new plan for the remaining steps based on the current state and potential previous error."""
        if not self.gemini_model:
            logger.warning("LLM not available, cannot recalculate plan.")
            return None

        error_context = ""
        if last_error:
            error_context += f"\nIMPORTANT CONTEXT: The previous action attempt failed.\n"
            if failed_action:
                error_context += f"Failed Action Details: {json.dumps(failed_action)}\n"
            error_context += f"Error Message: {last_error}\n"
            error_context += f"Please analyze this error and the current page state to create a recovery plan.\n"

        try:
            start_time = time.time()

            prompt = f"""
            You are a web automation assistant revising a plan after a potential failure.
            The original high-level plan was: {original_plan}
            We have completed steps up to index {current_step_index - 1}.
            The current step we were trying to achieve is: {original_plan[current_step_index]}
            The remaining original steps are: {original_plan[current_step_index:]}
            {error_context}
            CURRENT WEB PAGE STATE:
            URL: {state.get("url", "unknown")}
            TITLE: {state.get("title", "unknown")}
            CONTENT EXCERPT (first 1500 chars): {state.get("content", "")[:1500]}

            Based on the current page state, and considering the previous error (if any), create a revised, concise step-by-step plan (2-5 steps) to achieve the goal of the *remaining original steps*, starting from the current page. The plan should attempt to recover from the error if possible (e.g., by finding an alternative element, waiting longer, or trying a different approach).
            Focus on actionable steps from the current state.
            Return ONLY a JSON array of strings representing the new steps.
            """

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "response_mime_type": "application/json"
                }
            )

            response_time_ms = (time.time() - start_time) * 1000
            self.telemetry.record_llm_call(
                prompt_size=len(prompt),
                response_size=len(response.text),
                response_time_ms=response_time_ms
            )

            new_steps = self._parse_json_response(response.text)
            if isinstance(new_steps, list) and all(isinstance(step, str) for step in new_steps) and new_steps:
                logger.info(
                    f"Recalculated remaining steps considering error '{last_error}': {new_steps}")
                return new_steps
            else:
                logger.error(
                    f"LLM plan recalculation failed to return a valid non-empty list: {new_steps}")
                self.telemetry.record_error(
                    "plan_recalculation_failure",
                    "Failed to recalculate plan",
                    {"error": last_error, "response": response.text}
                )
                return None
        except Exception as e:
            logger.error(f"Error during plan recalculation: {e}")
            self.telemetry.record_error(
                "plan_recalculation_exception",
                str(e),
                {"error": last_error}
            )
            return None

    def _parse_json_response(self, text: str) -> Any:
        """Parse JSON response from LLM, handling different JSON formatting"""
        try:
            # Clean up the text to extract just the JSON part
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            # Parse and return the JSON
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            self.telemetry.record_error(
                "json_parse_error",
                str(e),
                # Record first 500 chars of text for debugging
                {"text": text[:500]}
            )
            return None

    def get_telemetry_report(self) -> Dict[str, Any]:
        """Return the current telemetry report with all metrics"""
        return self.telemetry.get_metrics_report()
