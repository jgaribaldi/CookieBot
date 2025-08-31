from pprint import pprint
from typing import List

from context import RecipyContext, RecipyData, json_to_recipy_data
from llm import Llm


class CookieBotAgent:
    def __init__(self, recipy_context: RecipyContext):
        self._recipy_context = recipy_context

    def process_request(self, request: str) -> List[str]:
        recipies = self._get_recipies(request)
        return [recipy.name for recipy in recipies]

    def _get_recipies(self, request: str) -> List[RecipyData]:
        recipies = self._recipy_context.find_similar(request, 10)
        if recipies:
            return [json_to_recipy_data(recipy) for recipy in recipies]
        else:
            return []


if __name__ == "__main__":
    print("Hello CookieBot")

    recipy_context = RecipyContext()
    recipy_context.load_or_create_embeddings(
        "data/embeddings.pickle", "data/recetasdelaabuela.csv"
    )

    available_ingredients = ["huevos", "cebollas", "pescado"]
    agent = CookieBotAgent(recipy_context)
    recipies = agent.process_request(
        f"Los ingredientes que tengo son: {", ".join(available_ingredients)}"
    )

    llm = Llm(url="http://localhost:11434", sys_prompt_filename="system_prompt.txt")
    selected_recipy = llm.ask_recipy(
        ingredients=available_ingredients, recipies=recipies
    )
    print(selected_recipy)
