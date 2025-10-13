# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple
from .base import QAPIABuilder


class EmailIPIABuilder(QAPIABuilder):
    name = "email"
    system_prompt_template = (
        "You are an email assistant responsible for assisting the user in processing and analyzing emails. "
        "Please fulfill the user's requests based on the content of the email. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the email{guidance}:\n{context}"
    )
    
    system_prompt_aside = (
        "You are an email assistant responsible for assisting the user in processing and analyzing emails. "
        "Please fulfill the user's requests based on the content of the email. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Question: {question}"
    )
    data_aside = "Email Content: {context}"
    
    user_prompt_template = [
        "Question: {question}",
        (
            "You are an email assistant responsible for processing and analyzing emails. "
            "Please answer my question based on the content of the email. "
            "Your answers should be as concise as possible.\n\n"
            "Here is the content of the email{guidance}:\n{context}\n\n"
            "Question: {question}"
        ),
    ]

    def construct_prompt(self, example: Any) -> Tuple[str, str]:
        system_prompt = self.system_prompt_aside.format(question=example["question"])
        data_aside = self.data_aside.format(context=example["context"])
        return system_prompt, data_aside

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"]

        if ideal != "unkown":
            return f"Answer: {ideal}."
        else:
            return "There is no information about the question in the email."
