import json
from typing import List

from context import RecipyData, RecipyStorage, json_to_recipy_data
from llm import Llm


class CookieBotAgent:
    def __init__(self, recipy_storage: RecipyStorage):
        self._recipy_storage = recipy_storage

    def process_request(self, request: str) -> List[str]:
        recipies = self._get_recipies(request)
        return [
            json.dumps({"name": recipy.name, "ingredients": recipy.ingredients})
            for recipy in recipies
        ]

    def _get_recipies(self, request: str) -> List[RecipyData]:
        recipies = self._recipy_storage.find_similar(request, 10)
        if recipies:
            return [json_to_recipy_data(recipy) for recipy in recipies]
        else:
            return []


if __name__ == "__main__":
    print("Hello CookieBot")
    create_embeddings = False
    available_ingredients = ["huevos", "cebollas", "merluza"]

    recipy_storage = RecipyStorage(file_path="data/recipystorage.chroma")

    if create_embeddings:
        recipy_storage.add_data("data/recetasdelaabuela.csv")
        print("Finsihed adding data")

    agent = CookieBotAgent(recipy_storage)
    query = f"Los ingredientes que tengo son: {", ".join(available_ingredients)}"
    recipies = agent.process_request(query)

    if recipies:
        llm = Llm(sys_prompt_filename="system_prompt.txt")
        selected_recipy = llm.ask_recipy(
            ingredients=available_ingredients, recipies=recipies
        )
        print(selected_recipy)
