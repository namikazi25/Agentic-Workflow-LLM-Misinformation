# scratch/step04_webqa_stub.py
"""
Step-04: stub Web-QA module that relies on LLM knowledge only.
Replace with real Brave search later.
"""

from __future__ import annotations
from typing import Tuple

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate


class WebQAModuleStub:
    """Tiny wrapper around LLM (no external search)."""

    def __init__(self, question: str, llm) -> None:
        self.question = question
        self.llm = llm

    def run(self) -> Tuple[str, bool]:
        if not self.question:
            return "Empty question", False

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based on your knowledge."),
            ("human", self.question)
        ])
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            answer = chain.invoke({}).get("text", "").strip()
            return answer, bool(answer)
        except Exception as e:
            return f"WebQA Error: {e}", False