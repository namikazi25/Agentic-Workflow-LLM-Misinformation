# scratch/step03_qgen_no_enrich.py  (LCEL + no literal braces)
from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple

from langchain_core.prompts import ChatPromptTemplate


class QAGenerationToolNoEnrich:
    def __init__(
        self,
        headline: str,
        previous_qa: List[Dict[str, str]] | None = None,
    ) -> None:
        self.headline = headline
        self.previous_qa = previous_qa or []

    def run(self, llm) -> Tuple[str, bool]:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are a fact-checking question generator. "
                "Given the HEADLINE below, produce ONE concise, web-searchable question "
                "to help verify it. Avoid repeating questions already asked.\n\n"
                "Previous Q-A (if any):\n{previous_qa}"),
            ("human", "HEADLINE: {headline}"),  # ← only {headline} is a variable
        ])

        chain = prompt | llm
        try:
            response = chain.invoke({"headline": self.headline, "previous_qa": json.dumps(self.previous_qa, indent=2)})
            question = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return question, bool(question)
        except Exception as e:
            return f"Q-Gen Error: {e}", False