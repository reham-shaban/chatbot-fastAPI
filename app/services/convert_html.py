import os, json
from bs4 import BeautifulSoup
from langchain.schema import Document

class ConvertHTMLPipeline:
    """
    A pipeline for converting HTML content into structured JSON data and further processing it 
    into document objects. The pipeline provides methods for extracting data from HTML tables, 
    paragraphs, lists, and headers, and converting this data into JSON or Document objects.
    """

    def __init__(self):
        """Initializes the ConvertHTMLPipeline class."""
        pass
    
    def _extract_table_data(self, table):
        """
        Extracts data from an HTML table and returns it as a list of lists.

        Args:
            table (bs4.element.Tag): The HTML table element to extract data from.

        Returns:
            list: A list of lists where each inner list represents a row in the table.
        """
        table_content = []
        rows = table.find_all('tr')
        for row in rows:
            row_data = []
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                row_data.append(cell_text)
            table_content.append(row_data)
        return table_content
    
    def _extract_post_content(self, post_div):
        """
        Extracts content from a post division, including headers, paragraphs, lists, and tables.

        Args:
            post_div (bs4.element.Tag): The HTML div element containing the post content.

        Returns:
            dict: A dictionary containing the extracted content, with keys 'header' for the header 
            text and 'text' for the list of content elements.
        """
        content = {}
        header_tag = post_div.find('h3')
        content['header'] = header_tag.get_text(strip=True) if header_tag else "No header"
        
        content['text'] = []
        processed_texts = set()
        
        for element in post_div.find_all(['p', 'ul', 'ol', 'li', 'table']):
            if element.name in ['ul', 'ol']:
                list_items = []
                for li in element.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if item_text not in processed_texts:
                        list_items.append(item_text)
                        processed_texts.add(item_text)
                if list_items:
                    content['text'].append({'type': 'list', 'content': list_items})
            elif element.name == 'p':
                paragraph_text = element.get_text(strip=True)
                if paragraph_text not in processed_texts:
                    content['text'].append({'type': 'paragraph', 'content': paragraph_text})
                    processed_texts.add(paragraph_text)
            elif element.name == 'table':
                table_data = self._extract_table_data(element)
                for row in table_data:
                    for cell in row:
                        processed_texts.add(cell.strip())
                content['text'].append({'type': 'table', 'content': table_data})
        
        return content
    
    def _html_to_json_v1(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all divs with class 'post'
        posts = soup.find_all('div', class_='post')

        # Convert all posts to JSON
        data = []
        for post in posts:
            post_content = self._extract_post_content(post)
            data.append(post_content)

        # Convert to JSON with ensure_ascii=False to avoid escaping characters
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        return json_data
    
    def _html_to_json(self, html_content):
        """
        Converts HTML content to a JSON string representing the structured content.

        Args:
            html_content (str): The HTML content to be converted.

        Returns:
            str: A JSON string representing the structured content.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        result = []

        # Process each section in the HTML
        sections = soup.find_all(['h1', 'h2', 'h3', 'div', 'p', 'table', 'ul'])
        current_section = {"header": "No header", "text": []}

        for section in sections:
            if section.name in ['h1', 'h2', 'h3']:
                # Start a new section if a header is found
                if current_section["text"]:
                    result.append(current_section)
                    current_section = {"header": section.get_text(strip=True), "text": []}
                else:
                    current_section["header"] = section.get_text(strip=True)
            elif section.name == 'p':
                current_section["text"].append({
                    "type": "paragraph",
                    "content": section.get_text(strip=True)
                })
            elif section.name == 'table':
                table_data = []
                for row in section.find_all('tr'):
                    row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    table_data.append(row_data)
                current_section["text"].append({
                    "type": "table",
                    "content": table_data
                })
            elif section.name == 'ul':
                # Handling unordered lists
                list_items = [li.get_text(strip=True) for li in section.find_all('li')]
                current_section["text"].append({
                    "type": "list",
                    "content": list_items
                })
    
        # Append the last section
        if current_section["text"]:
            result.append(current_section)

        # Return as JSON
        return json.dumps(result, ensure_ascii=False, indent=4)

    def convert_html_file_to_json(self, html_file_path):
        """
        Converts an HTML file to a JSON file by extracting and structuring the content.

        Args:
            html_file_path (str): The file path to the HTML file to be converted.

        Returns:
            str: The file path to the generated JSON file.

        Raises:
            FileNotFoundError: If the specified HTML file does not exist.
        """
        if not os.path.exists(html_file_path):
            raise FileNotFoundError(f"The file '{html_file_path}' does not exist.")
    
        # Read the HTML file with UTF-8 encoding
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Convert HTML to JSON
        json_data = self._html_to_json(html_content)

        # Construct the JSON file path
        base_name = os.path.splitext(os.path.basename(html_file_path))[0]
        json_file_path = os.path.join(os.path.dirname(html_file_path), base_name + '.json')

        # Write JSON to a file with UTF-8 encoding
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)

        print(f"Converted HTML to JSON and saved to {json_file_path}")

        return json_file_path
 
    def _load_and_split_json(self, file_path):
        """
        Loads a JSON file and splits its content into manageable chunks.

        Args:
            file_path (str): The file path to the JSON file.

        Returns:
            list: A list of chunks where each chunk is a portion of the JSON content.
        """
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for entry in data:
                    chunks.append(entry)
        except Exception as e:
            print(f"An error occurred while loading or splitting the JSON file: {e}")
        return chunks
    
    def _create_documents_from_chunks(self, table_chunks, my_metadata):
        """
        Creates Document objects from chunks of data with associated metadata.

        Args:
            table_chunks (list): The list of data chunks to be converted into Document objects.
            my_metadata (dict): Metadata to be associated with each Document.

        Returns:
            list: A list of Document objects.
        """
        documents = []
        for i, chunk in enumerate(table_chunks):
            content = json.dumps(chunk, ensure_ascii=False, indent=2)
            document = Document(page_content=content, metadata=my_metadata)
            documents.append(document)
        return documents

    def convert_json_to_documents(self, file_path, metadata):
        """
        Loads a JSON file, splits it into chunks, and converts it into a list of Document objects.

        Args:
            file_path (str): The file path to the JSON file.
            metadata (dict): Metadata to be associated with each Document.

        Returns:
            list: A list of Document objects created from the JSON content.
        """
        chunks = self._load_and_split_json(file_path)
        documents = self._create_documents_from_chunks(chunks, metadata)
        return documents
