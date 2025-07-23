# scratch/step05_qa_selector.py
"""
Step-05: pick the single best Q-A pair from a branch
for downstream final classifier.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate


class QASelector:
    """
    Given a list of Q-A pairs from ONE branch, choose the single most useful one.
    """

    def __init__(self, qa_pairs: List[Dict[str, Any]]) -> None:
        self.qa_pairs = qa_pairs

    def run(self, llm) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.qa_pairs:
            return None, True  # nothing to pick

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a fact-checking analyst. Below are several question-answer "
             "pairs generated to verify a news headline.\n"
             "Choose the **single most informative and relevant** pair that "
             "best helps determine whether the headline is true or false.\n"
             "Return ONLY the chosen pair in exactly this format:\n"
             "Question: <chosen question>\n"
             "Answer: <chosen answer>"),
            ("human",
             "Q-A pairs:\n" +
             "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.qa_pairs]))
        ])

        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            resp = chain.invoke({}).get("text", "").strip()

            # crude parsing
            q_match = resp.find("Question:")
            a_match = resp.find("Answer:")
            if q_match == -1 or a_match == -1:
                # fallback: pick first
                return self.qa_pairs[0], False

            question = resp[q_match + 9:a_match].strip()
            answer = resp[a_match + 7:].strip()
            return {"question": question, "answer": answer}, True

        except Exception as e:
            return {"question": "Selector failed", "answer": str(e)}, False