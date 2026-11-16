try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF releases
    import fitz
import spacy as sp

class DataIngestion(object):
    def __init__(self):
        """Initialize the PDF ingestion helper."""

        pass

    def get_sentences(self):
        """Create the lightweight spaCy sentence-segmentation pipeline."""

        """Split text into sentences"""
        nlp = sp.blank("en")
        nlp.add_pipe("sentencizer")

        return nlp

    def text_formatter(self,pdf_text:str):
        """Normalize line breaks and repeated whitespace in extracted PDF text."""

        """Performs an initial formatting"""
        cleaned_text = " ".join(pdf_text.replace("\n", " ").split())

        return cleaned_text

    def read_with_outline(self,pdf_path):
        """Extract PDF text chapter by chapter using its table of contents."""

        """
        Opens a PDF file and reads chapters following the table of content.
        
        RETURNS: list of dicts as {"chapter","chapter_title","text"}
        """

        pdf = fitz.open(pdf_path)
        outline = pdf.get_toc()             # list of [level, title, page number (starting from 1)]
        if not outline:
            outline = [[1,"Full text",1]]

        chapters = []
        for i,(level,title,page_number) in enumerate(outline):
            first_page = page_number - 1                                # first page of the chapter
            last_page = (
                outline[i+1][2] - 1 
                if i+1 < len(outline) else len(pdf) - 1
            )                                                           # last page of the chapter
            
            last_page = min(max(first_page,last_page),len(pdf) - 1)     # chapter could end where next starts

            chapter_text = "".join(
                pdf[p].get_text() 
                for p in range(first_page,last_page+1)
            )
            chapter_text = self.text_formatter(chapter_text)

            chapters.append(
                {"chapter":i,"chapter_title":title,"text":chapter_text}
                )

        return chapters
