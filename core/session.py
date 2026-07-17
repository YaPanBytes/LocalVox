"""
Tracks conversation history for the current session so the LLM has
context across turns.
"""


class Session:
    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add_user_turn(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_turn(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        # Keep only the last N turns to bound LLM context/latency
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_history(self) -> list:
        return self.history
