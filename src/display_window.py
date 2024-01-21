import tkinter as tk
from pdf_search import PDFSearch
from collections import OrderedDict


class DisplayWindow(tk.Tk):
    def __init__(self, pdf_crawler: PDFSearch, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self._button_width = 10
        self._button_height = 1
        self._color_args = {"foreground": "white", "background": "black"}
        self._current_page = 0
        self._n_current_result = -1
        self._search_results = None
        self._pdf_crawler = pdf_crawler
        self._pdf_content_dict = pdf_crawler.get_pdf()
        self._frame_dict = OrderedDict()
        self._button_dict = OrderedDict()
        self._search_dict = OrderedDict()
        self._text_box_dict = OrderedDict()
        self.configure(background="blue")
        self._render()

    def _render(self):
        self._create_frames()
        self._create_buttons()
        self._create_search_box()
        self._create_text_box()
        self._pack_items()

    def _back_page(self):
        if self._current_page > 0:
            self._current_page -= 1
        self._update_text_box()

    def _create_buttons(self):
        button_args = self._get_button_args()
        button_args["master"] = self._frame_dict["button"]
        self._button_dict["back"] = tk.Button(
            text="Previous Page", command=self._back_page, **button_args
        )
        self._button_dict["label"] = tk.Label(
            master=self._frame_dict["button"],
            text=f"Page: {self._current_page + 1}/{len(self._pdf_content_dict)}",
        )
        self._button_dict["next"] = tk.Button(
            text="Next Page", command=self._next_page, **button_args
        )

    def _create_frames(self):
        self._frame_dict["label"] = tk.Frame(master=self)
        self._frame_dict["button"] = tk.Frame(master=self)
        self._frame_dict["search"] = tk.Frame(master=self)

    def _create_search_box(self):
        self._search_dict["search_box"] = tk.Text(
            master=self._frame_dict["search"], height=2, **self._color_args
        )
        self._search_dict["search_button"] = tk.Button(
            master=self._frame_dict["search"], command=self._search_press, text="Search"
        )
        self._search_dict["back_button"] = tk.Button(
            master=self._frame_dict["search"], command=self._decrement_result, text="<"
        )
        self._search_dict["forward_button"] = tk.Button(
            master=self._frame_dict["search"], command=self._increment_result, text=">"
        )

    def _create_text_box(self):
        # main text box not in frame so can fill window #
        text_edit_box = tk.Text(master=self, **self._color_args)
        text_edit_box.grid_columnconfigure(0, weight=1)
        text_edit_box.insert(tk.END, self._pdf_content_dict[str(self._current_page)])
        text_edit_box.tag_configure(
            "start", background="OliveDrab1", foreground="black"
        )
        text_edit_box.config(state=tk.DISABLED)
        self._text_box_dict["content"] = text_edit_box

    def _decrement_result(self):
        if self._search_results is None:
            self._search_press()
            self._n_current_result = 0
        else:
            self._n_current_result -= 1
        self._n_current_result = (
            0 if self._n_current_result < 0 else self._n_current_result
        )
        self._highlight_result()

    def _get_button_args(self):
        button_args = {"width": self._button_width, "height": self._button_height}
        button_args.update(**self._color_args)
        return button_args

    def _highlight_result(self):
        curr_result = self._search_results[self._n_current_result]
        page_str = curr_result["id"]
        text_to_highlight = curr_result["metadata"]
        page_num = int(page_str.split("_")[0])
        self._current_page = page_num
        self._update_text_box(text_to_highlight=text_to_highlight)

    def _increment_result(self):
        if self._search_results is None:
            self._search_press()
            self._n_current_result = 0
        else:
            self._n_current_result += 1
        self._highlight_result()

    def _next_page(self):
        if self._current_page < len(self._pdf_content_dict) - 1:
            self._current_page += 1
        self._update_text_box()

    def _pack_items(self):
        """
        Pack all items stored in dictionaries
        :return: None
        """
        # ns = vertical fill, ew = horizontal fill #
        col_iterate_button = 0
        align_const = tk.TOP
        for button_name, curr_button in self._button_dict.items():
            curr_button.grid(row=0, column=col_iterate_button, sticky="ew")
            col_iterate_button += 1
        for search_obj_name, search_obj in self._search_dict.items():
            search_obj.pack(side=tk.LEFT)
        for frame_name, curr_frame in self._frame_dict.items():
            curr_frame.pack(side=align_const)
        for text_box_name, curr_text_box in self._text_box_dict.items():
            curr_text_box.pack(expand=True, fill="both")

    def _search_press(self):
        search_box = self._search_dict["search_box"]
        search_str = search_box.get("1.0", tk.END)
        self._search_results = self._pdf_crawler.search(string=search_str)
        self._n_current_result = 0
        self._highlight_result()

    def _update_text_box(self, text_to_highlight: str = None):
        text_edit_box = self._text_box_dict["content"]
        page_label = self._button_dict["label"]
        page_label.configure(
            text=f"Page: {self._current_page + 1}/{len(self._pdf_content_dict)}"
        )
        text_edit_box.config(state=tk.NORMAL)
        text_edit_box.delete("1.0", "end")

        # render matching page to get index #
        page_contents = self._pdf_content_dict[str(self._current_page)]
        text_edit_box.insert(tk.END, page_contents)

        if text_to_highlight is not None:
            start_idx_combined = text_edit_box.search(
                text_to_highlight, "1.0", stopindex="end"
            )
            start_line, start_char = [int(x) for x in start_idx_combined.split(".")]
            text_to_highlight_split = text_to_highlight.split("\n")
            n_highlight_lines = len(text_to_highlight_split)
            end_char = len(text_to_highlight_split[-1])
            n_highlight_lines = (
                n_highlight_lines - 1 if n_highlight_lines > 1 else n_highlight_lines
            )
            end_line = start_line + n_highlight_lines
            end_idx_combined = f"{end_line}.{end_char}"
            text_edit_box.tag_add("start", start_idx_combined, end_idx_combined)
        text_edit_box.pack(expand=True, fill="both")
        text_edit_box.config(state=tk.DISABLED)
