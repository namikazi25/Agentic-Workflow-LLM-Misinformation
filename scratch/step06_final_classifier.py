from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

from langchain_core.prompts import ChatPromptTemplate

class FinalClassifier:
    """
    Final veracity classifier.
    Input:
      - headline
      - image_relevancy_text
      - qa_pairs (list of best Q-A dicts, one per chain)
    Output:
      (decision, explanation)
    """

    SYSTEM_TEMPLATE = (
        "You are a senior misinformation-detection expert. Evaluate the headline below "
        "based STRICTLY on the provided evidence.\n\n"
        "Evidence:\n"
        "- Image–headline relevancy check: {relevancy}\n"
        "- Fact-checking Q&A pairs:\n{qa_context}\n\n"
        "Guidelines:\n"
        "- If the Q&A evidence **directly refutes** the headline, return **Misinformation**.\n"
        "- If it **supports** the headline, return **Not Misinformation**.\n"
        "- If evidence is **inconclusive**, return **Uncertain**.\n"
        "Ignore the image relevancy unless it clearly contradicts the Q&A.\n\n"
        "Reply in exactly this format:\n"
        "DECISION: <Misinformation|Not Misinformation|Uncertain>\n"
        "REASON: <concise paragraph citing evidence>"
    )

    def __init__(
        self,
        headline: str,
        relevancy_text: str,
        qa_pairs: List[Dict[str, Any]],
    ) -> None:
        self.headline = headline
        self.relevancy_text = relevancy_text
        self.qa_pairs = qa_pairs

    def run(self, llm) -> Tuple[str, str]:
        qa_context = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.qa_pairs
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_TEMPLATE),
            ("human", "Headline: {headline}"),
        ])

        # Pass all variables used in the prompt
        response = (prompt | llm).invoke({
            "headline": self.headline,
            "relevancy": self.relevancy_text,
            "qa_context": qa_context,
        })

        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Simple parser for output format
        dec_match = re.search(r"DECISION:\s*(Misinformation|Not Misinformation|Uncertain)", raw, re.I)
        reason_match = re.search(r"REASON:\s*(.+)", raw, re.I | re.S)
        decision = dec_match.group(1).strip() if dec_match else "Uncertain"
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        return decision, reason