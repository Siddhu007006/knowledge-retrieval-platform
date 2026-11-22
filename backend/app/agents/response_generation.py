"""Response Generation Agent: generate and verify answers against retrieved evidence."""
from app.generation.generator import generate_answer
from app.generation.grounding import check_grounding
from app.models.workflow import ContextPackage, GroundingResult


class ResponseGenerationAgent:
    def generate(self, context: ContextPackage) -> str:
        return generate_answer(context)

    def verify(self, context: ContextPackage, answer: str) -> GroundingResult:
        return check_grounding(context, answer)
