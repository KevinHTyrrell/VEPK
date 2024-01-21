from src.display_window import DisplayWindow
from src.sgpt_embedder import SGPTEmbedder
from src.pdf_search import PDFSearch


if __name__ == "__main__":
    write_cache = False
    read_cache = True
    chars_to_skip = 0
    min_length_cutoff = 20
    pdf_filepath = "data/astronomy_openstax.pdf"
    char_replace_filepath = "ref/char_replace.yml"

    embedder = SGPTEmbedder()
    embedding_dims = embedder.get_sentence_embedding_dimension()

    pdf_search_args = {
        "pdf_filepath": pdf_filepath,
        "char_config_filepath": char_replace_filepath,
        "embedder": embedder,
        "min_sentence_length": 20,
        "n_chars_skip": chars_to_skip,
        "load_from_cache": True,
        "save_to_cache": True,
    }
    pdf_crawler = PDFSearch(**pdf_search_args)
    pdf_crawler.config()

    main_window = DisplayWindow(pdf_crawler=pdf_crawler)
    main_window.mainloop()
