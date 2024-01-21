from sentence_transformers import SentenceTransformer


class SGPTEmbedder(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: str = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
        *args,
        **kwargs
    ):
        super().__init__(model_name_or_path=model_name_or_path, *args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(
            len(self._first_module().tokenizer)
        )
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode(
            "[SOS]", add_special_tokens=False
        )[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode(
            "{SOS}", add_special_tokens=False
        )[0]
        self._first_module().bos_spec_token_q_rep = (
            self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        )
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode(
            "]", add_special_tokens=False
        )[0]
        self._first_module().bos_spec_token_d_rep = (
            self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        )
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode(
            "}", add_special_tokens=False
        )[0]
        self._first_module().replace_bos = True

    def encode(self, sentences, **kwargs):
        is_query = kwargs.pop("is_query", True)
        sos_token = "[SOS]" if is_query else "{SOS}"
        sentences = (
            sos_token + sentences
            if isinstance(sentences, str)
            else [sos_token + sent for sent in sentences]
        )
        return super().encode(sentences, **kwargs)
