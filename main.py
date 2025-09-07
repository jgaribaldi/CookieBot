import json
from typing import List, Optional

from context import RecipyData, RecipyStorage, json_to_recipy_data
from llm import Llm


class CookieBotAgent:
    def __init__(self, recipy_storage: RecipyStorage, llm: Llm):
        self._recipy_storage = recipy_storage
        self._llm = llm

    def process_request(self, available_ingredients: List[str]) -> Optional[str]:
        recipies = self._get_recipies(available_ingredients)
        if recipies:
            llm_data = [
                json.dumps({"name": recipy.name, "ingredients": recipy.ingredients})
                for recipy in recipies
            ]
            if llm_data:
                return llm.ask_recipy(
                    ingredients=available_ingredients, recipies=llm_data
                )
            else:
                return None
        else:
            return None

    def _get_recipies(self, available_ingredients: List[str]) -> List[RecipyData]:
        query = f"Los ingredientes que tengo son: {", ".join(available_ingredients)}"
        recipies = self._recipy_storage.find_similar(query, 10)
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

    llm = Llm(sys_prompt_filename="system_prompt.txt")

    agent = CookieBotAgent(recipy_storage, llm)
    suggested_recipy = agent.process_request(available_ingredients)
    print(suggested_recipy)
