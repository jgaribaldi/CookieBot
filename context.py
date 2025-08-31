import json
from typing import List, Optional

import chromadb
import pandas
from chromadb.utils.batch_utils import create_batches
from torch import Tensor


class DataToSave:
    def __init__(self, embeddings: List[Tensor], texts: List[str]) -> None:
        self.embeddings = embeddings
        self.texts = texts


class RecipyData:
    def __init__(self, name: str, ingredients: str, steps: str) -> None:
        self.name = name
        self.ingredients = ingredients
        self.steps = steps


class RecipyStorage:
    def __init__(self, file_path: str):
        self._client = chromadb.PersistentClient(path=file_path)
        self._collection = self._client.get_or_create_collection(
            name="recipies",
            metadata={
                "description": "set of recipies in Spanish",
            },
        )

    def find_similar(self, query: str, top_k: int) -> Optional[List[str]]:
        results = self._collection.query(query_texts=[query], n_results=top_k)
        if results:
            documents = results["documents"]
            if documents:
                return documents[0]
        return None

    def add_data(self, raw_data_file: str) -> None:
        self._create_embeddings(raw_data_file)

    def _create_embeddings(self, raw_data_file: str) -> None:
        print("Creating embeddings from raw data...")
        input_df = pandas.read_csv(raw_data_file, encoding="utf-8")
        ids = input_df.apply(
            lambda row: str(row["Id"]),
            axis=1,
        ).to_list()
        documents = input_df.apply(
            lambda row: _row_to_json(
                row["Nombre"],
                row["Ingredientes"],
                row["Pasos"],
            ),
            axis=1,
        ).to_list()
        metadatas = input_df.apply(
            lambda row: {"ingredients": row["Ingredientes"]},
            axis=1,
        ).to_list()
        batches = create_batches(
            api=self._client, ids=ids, documents=documents, metadatas=metadatas
        )
        for batch in batches:
            print(f"Adding batch of size {len(batch[0])}")
            self._collection.add(documents=batch[3], ids=batch[0], metadatas=batch[2])


def _row_to_json(name: str, ingredients: str, steps: str) -> str:
    recipy_data = RecipyData(name, ingredients, steps)
    return json.dumps(recipy_data.__dict__)


def json_to_recipy_data(json_string: str) -> RecipyData:
    data = json.loads(json_string)
    return RecipyData(**data)
