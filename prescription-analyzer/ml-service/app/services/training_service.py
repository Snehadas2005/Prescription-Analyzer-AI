from typing import List, Dict, Any


class TrainingService:
    def __init__(self) -> None:
        # In-memory feedback storage for demo
        self._feedback_buffer: List[Dict[str, Any]] = []
        self._threshold: int = 10  # e.g. retrain after 10 feedbacks

    async def store_feedback(self, feedback: Dict[str, Any]) -> None:
        # TODO: save to DB / file in real implementation
        self._feedback_buffer.append(feedback)

    async def should_retrain(self) -> bool:
        # TODO: use real logic (time-based, count-based, performance-based)
        return len(self._feedback_buffer) >= self._threshold
