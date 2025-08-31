from typing import List, Optional

import ollama


class Llm:
    def __init__(self, url: str):
        self._client = ollama.Client(host=url)
        self._system_prompt = (
            "Context: you will analyze cooking recipies based on this input data:"
            "1. a list of available ingredients: a comma separated list of ingredients that are available to prepare the recipy"
            "2. a list of recipies that can be prepared with the given list of ingredients - This will be a string containing a list of json formatted strings containing recipy information (name, ingredients and steps)"
            "All the information will be in spanish"
            "For each recipy, you will have available and unavailable ingredients"
            "Task: determine which recipy of the input is the best to prepare - The best recipy to prepare is the one that maximizes the use of the available ingredients"
            "Criteria: you should utilize all the available ingredients - You can apply variations to a recipy if that involves utilizing an existing ingredient, even if the match is not exact"
            "Outcome: return a response in spanish that contains the recipy name, the list of available ingredients that will be used for the recipy, and the list of ingredients necessary to prepare the recipy but aren't available"
        )
        self._user_message_1 = "List of available ingredients: {ingredients}"
        self._user_message_2 = "List of recipies: {recipies}"

    def ask_recipy(self, ingredients: List[str], recipies: List[str]) -> Optional[str]:
        system_message = ollama.Message(role="system", content=self._system_prompt)
        user_message_1 = ollama.Message(
            role="user",
            content=self._user_message_1.format(ingredients=", ".join(ingredients)),
        )
        user_message_2 = ollama.Message(
            role="user",
            content=self._user_message_2.format(recipies=_format_recipies(recipies)),
        )
        response = self._client.chat(
            model="llama3.2", messages=[system_message, user_message_1, user_message_2]
        )
        return response.message.content


def _format_recipies(recipies: List[str]) -> str:
    result = ""
    for number, recipy in enumerate(recipies):
        result += f"Receta {number}: {recipy}\n"
    return result
