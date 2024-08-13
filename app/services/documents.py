from bs4 import BeautifulSoup
import json

def extract_post_content(post_div):
    content = {}

    # Extract the header
    header_tag = post_div.find('h3')
    content['header'] = header_tag.get_text(strip=True) if header_tag else "No header"

    # Extract paragraphs, lists, and tables
    content['text'] = []
    processed_texts = set()  # Track all texts that have been added to avoid duplicates

    for element in post_div.find_all(['p', 'ul', 'ol', 'li', 'table']):
        # For lists
        if element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li'):
                item_text = li.get_text(strip=True)
                if item_text not in processed_texts:
                    list_items.append(item_text)
                    processed_texts.add(item_text)  # Track this content
            if list_items:  # Only add the list if it's not empty
                content['text'].append({
                    'type': 'list',
                    'content': list_items
                })

        # For paragraphs
        elif element.name == 'p':
            paragraph_text = element.get_text(strip=True)
            if paragraph_text not in processed_texts:  # Add paragraph only if not already processed
                content['text'].append({
                    'type': 'paragraph',
                    'content': paragraph_text
                })
                processed_texts.add(paragraph_text)  # Track this content

        # For tables
        elif element.name == 'table':
            table_data = extract_table_data(element)
            # Flatten table content to compare with paragraphs
            for row in table_data:
                for cell in row:
                    processed_texts.add(cell.strip())  # Track each cell's text
            content['text'].append({
                'type': 'table',
                'content': table_data
            })

    return content

def extract_table_data(table):
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

def html_to_json(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all divs with class 'post'
    posts = soup.find_all('div', class_='post')

    # Convert all posts to JSON
    data = []
    for post in posts:
        post_content = extract_post_content(post)
        data.append(post_content)

    # Convert to JSON with ensure_ascii=False to avoid escaping characters
    json_data = json.dumps(data, ensure_ascii=False, indent=4)
    return json_data