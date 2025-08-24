from context import RecipyContext


class CookieBotAgent:
    def __init__(self, recipy_context: RecipyContext):
        self._recipy_context = recipy_context

    def process_request(self, request: str):
        context = self._recipy_context.find_similar(request, 10)
        if context:
            for result in context:
                print(result)
        else:
            print("No results")


if __name__ == "__main__":
    print("Hello CookieBot")

    recipy_context = RecipyContext()
    recipy_context.load_or_create_embeddings(
        "data/embeddings.pickle", "data/recetasdelaabuela.csv"
    )

    available_ingredients = "huevos, cebollas"

    agent = CookieBotAgent(recipy_context)
    recipy = agent.process_request(
        f"Los ingredientes que tengo son: {available_ingredients}"
    )
    print(recipy)
