# parser.py
from markitdown import MarkItDown

class MarkdownParser:
    def __init__(self, enable_plugins=False, llm_client=None, llm_model=None):
        self.md = MarkItDown(enable_plugins=enable_plugins, llm_client=llm_client, llm_model=llm_model)

    def convert_file(self, filepath: str) -> str:
        result = self.md.convert(filepath)
        return result.text_content
    def convert_string(self, content: str) -> str:
        result = self.md.convert_string(content)
        return result.text_content

    def convert_html(self, html_content: str) -> str:
        result = self.md.convert_html(html_content)
        return result.text_content

    def convert_markdown(self, markdown_content: str) -> str:
        result = self.md.convert_markdown(markdown_content)
        return result.text_content