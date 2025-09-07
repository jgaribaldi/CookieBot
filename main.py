import argparse
import json
from typing import List, Optional

from common import read_text_file
from context import RecipyData, RecipyStorage, json_to_recipy_data
from llm import Llm


class CookieBotAgent:
    def __init__(self, recipy_storage: RecipyStorage, llm: Llm, query_prompt_file: str):
        self._recipy_storage = recipy_storage
        self._llm = llm
        self._query_prompt_file = query_prompt_file

    def process_request(self, available_ingredients: List[str]) -> Optional[str]:
        recipies = self._get_recipies(available_ingredients)
        if recipies:
            llm_data = [
                json.dumps({"name": recipy.name, "ingredients": recipy.ingredients})
                for recipy in recipies
            ]
            if llm_data:
                return self._llm.ask_recipy(
                    ingredients=available_ingredients, recipies=llm_data
                )
            else:
                return None
        else:
            return None

    def _get_recipies(self, available_ingredients: List[str]) -> List[RecipyData]:
        query = read_text_file(self._query_prompt_file).format(
            ingredients=", ".join(available_ingredients)
        )
        recipies = self._recipy_storage.find_similar(query, 10)
        if recipies:
            return [json_to_recipy_data(recipy) for recipy in recipies]
        else:
            return []


def _parse_arguments() -> bool:
    parser = argparse.ArgumentParser(
        description="CookieBot: reads available ingredients and suggests recipies"
    )
    parser.add_argument(
        "-c",
        "--create-embeddings",
        help="Reads the input csv file and creates embeddings in ChromaDB",
        action="store_true",
    )
    args = parser.parse_args()
    return args.create_embeddings


def main() -> int:
    print("Hello CookieBot")
    available_ingredients = ["huevos", "cebollas", "merluza"]

    recipy_storage = RecipyStorage(file_path="data/recipies_db")

    if _parse_arguments():
        recipy_storage.add_data("data/recetasdelaabuela.csv")
        print("Finsihed adding data")
    else:
        llm = Llm(
            sys_prompt_filename="system_prompt.txt", language="Argentinian Spanish"
        )

        agent = CookieBotAgent(recipy_storage, llm, "query_prompt.txt")
        suggested_recipy = agent.process_request(available_ingredients)
        print(suggested_recipy)
    return 0


if __name__ == "__main__":
    exit(main())
