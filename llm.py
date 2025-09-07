import os
from typing import List, Optional

import anthropic
from anthropic.types import MessageParam

from common import read_text_file


class Llm:
    def __init__(self, sys_prompt_filename: str, language: str):
        self._client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        system_prompt = read_text_file(sys_prompt_filename).format(language=language)
        self._system_message = MessageParam(
            role="assistant",
            content=system_prompt,
        )
        self._user_message_1 = "List of available ingredients: {ingredients}"
        self._user_message_2 = "List of recipies: {recipies}"

    def ask_recipy(self, ingredients: List[str], recipies: List[str]) -> Optional[str]:
        user_message_1 = MessageParam(
            role="user",
            content=self._user_message_1.format(ingredients=", ".join(ingredients)),
        )
        user_message_2 = MessageParam(
            role="user",
            content=self._user_message_2.format(recipies=_format_recipies(recipies)),
        )
        response = self._client.messages.create(
            model="claude-sonnet-4-0",
            messages=[self._system_message, user_message_1, user_message_2],
            max_tokens=1024,
        )
        return response.content[0].text


def _format_recipies(recipies: List[str]) -> str:
    result = ""
    for number, recipy in enumerate(recipies):
        result += f"Receta {number}: {recipy}\n"
    return result
