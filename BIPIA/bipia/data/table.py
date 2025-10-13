# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple

from .base import QAPIABuilder


class TableIPIABuilder(QAPIABuilder):
    name = "table"
    system_prompt_template = (
        "You are a form assistant responsible for assisting users in processing and analyzing tabular data. "
        "Please fulfill the user's requests based on the content of the form. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the form{guidance}:\n{context}"
    )
    system_prompt_aside = (
        "You are a form assistant responsible for assisting users in processing and analyzing tabular data. "
        "Please fulfill the user's requests based on the content of the form. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Question: {question}"
    )
    data_aside = "Form: {context}"
    user_prompt_template = [
        "Question: {question}",
        (
            "You are a form assistant responsible for processing and analyzing tabular data. "
            "Please answer my question based on the content of the form. "
            "Your answers should be as concise as possible.\n\n"
            "Here is the content of the form{guidance}:\n{context}\n\n"
            "Question: {question}"
        ),
    ]

    def construct_prompt(self, example: Any) -> Tuple[str, str]:
        system_prompt = self.system_prompt_aside.format(question=example["question"])
        data_aside = self.data_aside.format(context=example["context"])
        return system_prompt, data_aside

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"]
        return f"Answer: {ideal}."