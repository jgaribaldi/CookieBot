import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas
from sentence_transformers import SentenceTransformer
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


class RecipyContext:
    def __init__(self) -> None:
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._loaded_embeddings = DataToSave(embeddings=[], texts=[])

    def find_similar(self, query: str, top_k: int = 1) -> Optional[List[str]]:
        similarities = self._calculate_similarities(query)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self._loaded_embeddings.texts[i] for i in top_indices]

    def load_or_create_embeddings(
        self, embeddings_file: str, raw_data_file: str
    ) -> None:
        file = Path(embeddings_file)

        if not file.exists():
            self._create_embeddings(raw_data_file, embeddings_file)
        else:
            file_empty = os.stat(embeddings_file).st_size == 0
            if file_empty:
                self._create_embeddings(raw_data_file, embeddings_file)

        self._loaded_embeddings = pandas.read_pickle(embeddings_file)

    def _create_embeddings(self, raw_data_file: str, embeddings_file: str) -> None:
        print("Creating embeddings from raw data...")
        input_df = pandas.read_csv(raw_data_file, encoding="utf-8")

        texts_to_embed = input_df["Ingredientes"].astype(str).to_list()
        embeddings = [self._model.encode(text) for text in texts_to_embed]

        texts_to_save = input_df.apply(
            lambda row: _row_to_json(
                row["Nombre"],
                row["Ingredientes"],
                row["Pasos"],
            ),
            axis=1,
        ).to_list()

        data_to_save = DataToSave(embeddings, texts_to_save)
        with open(embeddings_file, "wb") as f:
            pickle.dump(data_to_save, f)
        print("Embeddings ready to use!")

    def _calculate_similarities(self, query: str) -> List[float]:
        query_embedding = self._model.encode(query)
        query_array = np.array(query_embedding)
        loaded_embeddings_array = np.array(self._loaded_embeddings.embeddings)
        return np.dot(loaded_embeddings_array, query_array)


def _row_to_json(name: str, ingredients: str, steps: str) -> str:
    recipy_data = RecipyData(name, ingredients, steps)
    return json.dumps(recipy_data.__dict__)
