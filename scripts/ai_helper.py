"""
AI Helper - Wrapper for calling LLM CLIs (Gemini/Claude) with fallback strategy.

Based on patterns from /tmp/gemini/ project.
"""

import json
import subprocess
import sys
from typing import Dict, Optional, Tuple


class AIError(Exception):
    """Raised when AI call fails after all retries."""


def call_gemini(prompt: str, model: str = "gemini-2.5-flash", timeout: int = 60) -> str:
    """
    Call Gemini CLI with prompt via stdin.

    Args:
        prompt: Text prompt to send
        model: Gemini model name
        timeout: Timeout in seconds

    Returns:
        AI response text

    Raises:
        AIError: If gemini command fails
    """
    try:
        result = subprocess.run(
            ["gemini", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise AIError(f"Gemini CLI failed: {result.stderr}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise AIError(f"Gemini CLI timeout after {timeout}s")
    except FileNotFoundError:
        raise AIError("Gemini CLI not found. Install: npm install -g @google/generative-ai-cli")
    except Exception as e:
        raise AIError(f"Gemini call failed: {e}")


def call_claude(prompt: str, model: str = "haiku", timeout: int = 60) -> str:
    """
    Call Claude CLI with prompt via stdin.

    Args:
        prompt: Text prompt to send
        model: Claude model (haiku, sonnet, opus)
        timeout: Timeout in seconds

    Returns:
        AI response text

    Raises:
        AIError: If claude command fails
    """
    try:
        result = subprocess.run(
            ["claude", "--dangerously-skip-permissions", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise AIError(f"Claude CLI failed: {result.stderr}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise AIError(f"Claude CLI timeout after {timeout}s")
    except FileNotFoundError:
        raise AIError("Claude CLI not found. Install: https://github.com/anthropics/claude-cli")
    except Exception as e:
        raise AIError(f"Claude call failed: {e}")


def parse_json_response(response: str) -> Dict:
    """
    Parse JSON from AI response, handling markdown code blocks.

    Args:
        response: AI response text

    Returns:
        Parsed JSON dict

    Raises:
        AIError: If parsing fails
    """
    text = response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text.replace("```json\n", "", 1).replace("\n```", "", 1)
    elif text.startswith("```"):
        text = text.replace("```\n", "", 1).replace("\n```", "", 1)

    # Try to extract JSON object if surrounded by text
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise AIError(f"Failed to parse JSON: {e}\nResponse: {text[:500]}")


def call_ai(
    prompt: str,
    primary: str = "gemini",
    gemini_model: str = "gemini-2.5-flash",
    claude_model: str = "haiku",
    timeout: int = 60,
    retries: int = 1
) -> Tuple[str, str]:
    """
    Call AI with fallback strategy (primary → secondary).

    Args:
        prompt: Text prompt
        primary: Primary AI ('gemini' or 'claude')
        gemini_model: Gemini model name
        claude_model: Claude model name
        timeout: Timeout in seconds
        retries: Number of retries per AI

    Returns:
        (response_text, model_used)

    Raises:
        AIError: If all attempts fail
    """
    secondary = "claude" if primary == "gemini" else "gemini"
    errors = []

    # Try primary AI
    for attempt in range(retries):
        try:
            if primary == "gemini":
                response = call_gemini(prompt, gemini_model, timeout)
            else:
                response = call_claude(prompt, claude_model, timeout)

            return response, primary

        except AIError as e:
            errors.append(f"{primary} attempt {attempt+1}: {e}")

    # Fallback to secondary AI
    for attempt in range(retries):
        try:
            if secondary == "gemini":
                response = call_gemini(prompt, gemini_model, timeout)
            else:
                response = call_claude(prompt, claude_model, timeout)

            return response, secondary

        except AIError as e:
            errors.append(f"{secondary} attempt {attempt+1}: {e}")

    # All attempts failed
    raise AIError(f"All AI calls failed:\n" + "\n".join(errors))


def call_ai_json(
    prompt: str,
    primary: str = "gemini",
    retries: int = 2,
    **kwargs
) -> Tuple[Dict, str]:
    """
    Call AI and parse JSON response.

    Args:
        prompt: Text prompt (should request JSON output)
        primary: Primary AI ('gemini' or 'claude')
        retries: Number of retries
        **kwargs: Additional args for call_ai()

    Returns:
        (parsed_dict, model_used)

    Raises:
        AIError: If call or parsing fails
    """
    response, model = call_ai(prompt, primary=primary, retries=retries, **kwargs)

    try:
        parsed = parse_json_response(response)
        return parsed, model
    except AIError as e:
        # Retry with error feedback
        if retries > 0:
            retry_prompt = f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {e}\nPlease return valid JSON only."
            return call_ai_json(retry_prompt, primary=primary, retries=retries-1, **kwargs)
        raise


if __name__ == "__main__":
    # Test
    test_prompt = """Return JSON with your name and version.
Format: {"ai": "name", "version": "x.y"}"""

    try:
        result, model = call_ai_json(test_prompt)
        print(f"✓ AI test successful ({model})")
        print(f"  Response: {json.dumps(result, indent=2)}")
    except AIError as e:
        print(f"✗ AI test failed: {e}", file=sys.stderr)
        sys.exit(1)
