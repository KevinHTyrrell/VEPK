import json
import warnings
import numpy as np
import os
import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from src.misc import read_file
from src.misc.string_fns import clean_string
from src.vector_db import VectorDB
from sentence_transformers import SentenceTransformer


class PDFSearch:
    def __init__(
        self,
        pdf_filepath: str,
        char_config_filepath: str,
        cache_dir: str = "cache",
        embedding_dims: int = -1,
        embedder: SentenceTransformer = None,
        load_from_cache: bool = True,
        n_chars_skip: int = 0,
        save_to_cache: bool = True,
        sentence_join_char: str = " ",
        verbose: bool = True,
        use_raw_metadata: bool = True,
        word_split_char: str = " ",
        **kwargs,
    ):
        self._char_config_filepath = char_config_filepath
        self._embedder = embedder
        self._n_chars_skip = n_chars_skip
        self._pdf_filepath = pdf_filepath
        self._sentence_join_char = sentence_join_char
        self._verbose = verbose
        self._word_split_char = word_split_char
        self._use_raw_metadata = use_raw_metadata
        self._load_from_cache = load_from_cache
        self._save_to_cache = save_to_cache
        self._kwargs = kwargs
        self._filepath_join_char = os.path.join(" ", " ").strip()
        self._cache_filepath = os.path.join(
            cache_dir, pdf_filepath.split(self._filepath_join_char)[-1]
        )

        self._char_config = None
        self._embedded_dict = {}
        self._pdf_content_raw = None
        self._vector_index = VectorDB(
            embedding_dims
            if embedding_dims > 0
            else embedder.get_sentence_embedding_dimension()
        )

    def _check_for_cache(self):
        if not self._load_from_cache:
            return False
        if os.path.exists(self._cache_filepath) and os.path.exists(
            self._cache_raw_filepath
        ):
            with open(self._cache_filepath, "r") as file_reader:
                raw_data_dict = file_reader.read()
            with open(self._cache_raw_filepath, "r") as file_reader:
                raw_page_data = file_reader.read()
            try:
                self._embedded_dict = json.loads(raw_data_dict)
                self._pdf_content_raw = json.loads(raw_page_data)
                return True
            except:
                warnings.warn("CACHE NOT FOUND, BUILDING FROM SCRATCH")
                return False

    def _clean_and_label_sentences(self, char_replace_dict: dict):
        to_iterate = (
            tqdm(range(len(self._pdf_content_raw)))
            if self._verbose
            else range(len(self._pdf_content_raw))
        )
        for idx_page in to_iterate:
            selected_page = self._pdf_content_raw[str(idx_page)][self._n_chars_skip :]
            page_tokenized = sent_tokenize(selected_page)
            for idx_sentence in range(len(page_tokenized)):
                sentence_raw = page_tokenized[idx_sentence]
                word_list = [
                    s.lower().strip()
                    for s in sentence_raw.split(self._word_split_char)
                    if len(s.lower().strip()) > 0
                ]
                word_list_clean = [
                    clean_string(s, char_replace_dict) for s in word_list
                ]
                sentence_clean = self._sentence_join_char.join(word_list_clean)
                idx_combined = f"{idx_page}_{idx_sentence}"
                self._embedded_dict[idx_combined] = {
                    "sentence_raw": sentence_raw,
                    "sentence_clean": sentence_clean,
                }

    def _clean_cache_filepath(self):
        self._cache_filepath = (
            self._cache_filepath.split(".")[0]
            if "." in self._cache_filepath
            else self._cache_filepath
        )
        self._cache_raw_filepath = self._cache_filepath + "_raw"

    def _embed_strings(self, verbose: bool = True):
        if verbose:
            print("Embedding Strings")
        to_iterate = (
            tqdm(self._embedded_dict.items())
            if self._verbose
            else self._embedded_dict.items()
        )
        for idx, sentence_dict in to_iterate:
            sentence_clean = sentence_dict["sentence_clean"]
            sentence_raw = sentence_dict["sentence_raw"]
            sentence_embed = self._embedder.encode(sentence_clean).tolist()
            self._embedded_dict[idx] = {
                "sentence_clean": sentence_clean,
                "sentence_raw": sentence_raw,
                "embed": sentence_embed,
            }
        if self._save_to_cache:
            self._write_to_cache()

    def _load_vector_db(self, raw_metadata: bool = True):
        vector_emb = np.asarray([x["embed"] for x in self._embedded_dict.values()])
        vector_ids = [x for x in self._embedded_dict.keys()]
        metadata_to_use = "sentence_raw" if raw_metadata else "sentence_clean"
        vector_raw = [x[metadata_to_use] for x in self._embedded_dict.values()]
        self._vector_index.add_vectors(
            vectors=vector_emb, ids=vector_ids, metadata=vector_raw
        )

    def _read_char_config(self):
        self._char_config = read_file(self._char_config_filepath)

    def _read_pdf(self, verbose: bool = True):
        if verbose:
            print("Reading PDF")
        pdf_file_obj = open(self._pdf_filepath, "rb")
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        to_iterate = tqdm(pdf_reader.pages) if self._verbose else pdf_reader.pages
        page_content_dict = {}
        for i, page in enumerate(to_iterate):
            page_content_raw = page.extract_text()
            page_content_dict[str(i)] = page_content_raw[self._n_chars_skip :]
        self._pdf_content_raw = page_content_dict

    def _write_to_cache(self):
        if len(self._embedded_dict) == 0 or len(self._pdf_content_raw) == 0:
            warnings.warn("NO CONTENTS TO SAVE")
        else:
            with open(self._cache_filepath, "w") as file_writer:
                file_writer.write(json.dumps(self._embedded_dict))
            with open(self._cache_raw_filepath, "w") as file_writer:
                file_writer.write(json.dumps(self._pdf_content_raw))

    def config(self):
        self._clean_cache_filepath()
        if not self._check_for_cache():
            self._read_pdf()
            self._read_char_config()
            self._clean_and_label_sentences(self._char_config["replace"])
            self._embed_strings()
        self._load_vector_db(raw_metadata=self._use_raw_metadata)

    def search(self, string: str, k: int = 10, return_metadata: bool = True):
        if not self._vector_index.has_embedding_fn():
            embedded_string = self._embedder.encode(string)
            search_results = self._vector_index.get_neighbors(
                vector=embedded_string, k=k, return_metadata=return_metadata
            )
        else:
            search_results = self._vector_index.get_neighbors(
                string=string, k=k, return_metadata=return_metadata
            )
        return search_results

    def get_pdf(self):
        return self._pdf_content_raw

    def get_vector_db(self):
        return self._vector_index
