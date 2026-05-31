"""
System and user prompt templates for Claude.

Keeps prompts centralized and easily editable.
"""

SYSTEM_PROMPT = """\
You are a voice-controlled coding and task assistant.
Be concise.
If the user asks for multi-step work, give the plan first, then execute.
If the request is ambiguous, ask one clarifying question only when necessary.
Prefer actionable output.\
"""


def format_user_message(transcript: str) -> str:
    """
    Wrap the final voice transcript in a stable prompt format.

    Args:
        transcript: The finalized speech transcript.

    Returns:
        Formatted user message string for Claude.
    """
    return f"User voice transcript:\n{transcript}\n\nTask:\nRespond to the user's request."
