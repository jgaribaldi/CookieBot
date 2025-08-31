from typing import List, Optional

import ollama


class Llm:
    def __init__(self, url: str, sys_prompt_filename: str):
        self._client = ollama.Client(host=url)
        system_prompt = read_system_prompt(sys_prompt_filename)
        self._system_message = ollama.Message(role="system", content=system_prompt)
        self._user_message_1 = "List of available ingredients: {ingredients}"
        self._user_message_2 = "List of recipies: {recipies}"

    def ask_recipy(self, ingredients: List[str], recipies: List[str]) -> Optional[str]:
        user_message_1 = ollama.Message(
            role="user",
            content=self._user_message_1.format(ingredients=", ".join(ingredients)),
        )
        user_message_2 = ollama.Message(
            role="user",
            content=self._user_message_2.format(recipies=_format_recipies(recipies)),
        )
        response = self._client.chat(
            model="llama3.2",
            messages=[self._system_message, user_message_1, user_message_2],
        )
        return response.message.content


def _format_recipies(recipies: List[str]) -> str:
    result = ""
    for number, recipy in enumerate(recipies):
        result += f"Receta {number}: {recipy}\n"
    return result


def read_system_prompt(file_name: str) -> str:
    file = open("system_prompt.txt", "r")
    system_prompt = file.read()
    file.close()
    return system_prompt
