# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple

from .base import QAPIABuilder


class QAIPIADataset(QAPIABuilder):
    name = "qa"
    system_prompt_template = (
        "You are a question-and-answer assistant responsible for assisting the user in processing and analyzing news content. "
        "Please fulfill the user's requests based on the content of the news. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the news{guidance}:\n{context}"
    )
    system_prompt_aside = (
        "You are a question-and-answer assistant responsible for assisting the user in processing and analyzing news content. "
        "Please fulfill the user's requests based on the content of the news. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Question: {question}"
    )
    data_aside = "News: {context}"
    user_prompt_template = [
        "Question: {question}",
        (
            "You are a question-and-answer assistant responsible for processing and analyzing news content. "
            "Please answer my question based on the content of the news. "
            "Your answers should be as concise as possible.\n\n"
            "Here is the content of the news{guidance}:\n{context}\n\n"
            "Question: {question}"
        ),
    ]

    def construct_prompt(self, example: Any) -> Tuple[str, str]:
        system_prompt = self.system_prompt_aside.format(question=example["question"])
        data_aside = self.data_aside.format(context=example["context"])
        return system_prompt, data_aside
    
    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"][0]
        return f"Answer: {ideal}."
