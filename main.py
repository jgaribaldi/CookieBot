from typing import List

from context import RecipyData, RecipyStorage, json_to_recipy_data
from llm import Llm


class CookieBotAgent:
    def __init__(self, recipy_storage: RecipyStorage):
        self._recipy_storage = recipy_storage

    def process_request(self, request: str) -> List[str]:
        recipies = self._get_recipies(request)
        return [recipy.name for recipy in recipies]

    def _get_recipies(self, request: str) -> List[RecipyData]:
        recipies = self._recipy_storage.find_similar(request, 10)
        if recipies:
            return [json_to_recipy_data(recipy) for recipy in recipies]
        else:
            return []


if __name__ == "__main__":
    print("Hello CookieBot")

    available_ingredients = ["huevos", "cebollas", "pescado"]
    query = f"Los ingredientes que tengo son: {", ".join(available_ingredients)}"

    recipy_storage = RecipyStorage(file_path="data/recipystorage.chroma")
    # recipy_storage.add_data("data/recetasdelaabuela.csv")
    # print("Finsihed adding data")
    recipies = recipy_storage.find_similar(query, 10)

    agent = CookieBotAgent(recipy_storage)
    recipies = agent.process_request(query)

    if recipies:
        llm = Llm(url="http://localhost:11434", sys_prompt_filename="system_prompt.txt")
        selected_recipy = llm.ask_recipy(
            ingredients=available_ingredients, recipies=recipies
        )
        print(selected_recipy)
