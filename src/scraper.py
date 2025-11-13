# ----------------------------
# Project: Book Scraper - Scrape & Explore
# Author: Doufu (Github: DubuV2)
# Date: 2025-11-12
# Descrption: A script to scrape book data from a website to train, with options for detailed info, output formats
# ----------------------------

import requests, csv, logging, time, random, argparse, json, sqlite3, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- CONSTANTS ---
PROG_WITH_DETAILS = dict(PAGES=(0,30), LISTING=(30,50), DETAILS=(50,98), DONE=100)
PROG_NO_DETAILS = dict(PAGES=(0,30), LISTING=(30,98), DONE=100)

def _progress_range(done: int, total: int, start: int, end: int) -> int:
    """Calculates the percentage within a specified range."""
    if total == 0:
        return start
    frac = max(0.0, min(1.0, done / total))
    return int(start + frac * (end - start))

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Argument parsing
def parse_args():
    p = argparse.ArgumentParser(description="Scrape book data from a website.")
    p.add_argument('--url', type=str, default='http://books.toscrape.com/', help='Base URL of the books website')
    p.add_argument('--output', type=str, default='books.csv', help='Output CSV file path')
    p.add_argument('--delay', type=float, default=0.3, help='Delay between requests in seconds')
    p.add_argument('--details', action='store_true', help='Fetch detailed book information')
    p.add_argument('--max-workers', type=int, default=10, help='Number of threads for concurrent requests')
    p.add_argument('--format', type=str, choices=['csv', 'json', 'both'], default='csv', help='Output format')
    p.add_argument('--limit', type=int, default=0, help='Limit the number of books to scrape')
    p.add_argument('--db', default=None, help='Path to SQLite database file')   
    return p.parse_args()


def _human_bytes(size: int) -> str:
    """Convert a byte size into a human-readable format."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != 'B' else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def log_file_info(path: str, label: str = "") -> None:
    """Logs the absolute path and size of a file."""
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
        size = os.path.getsize(path)
        logging.info(f"{label}File saved: {abs_path} ({_human_bytes(size)})")
    except OSError as e:
        logging.warning(f"Error accessing file {path}: {e}")


def resolve_outputs(out: str, format: str):
    """"
    Resolves output file paths based on the specified format (CSV, JSON, or both).
    """
    base, ext = os.path.splitext(out)
    ext = ext.lower()

    if format == 'csv':
        return (out if ext == '.csv' else base + '.csv', None)
    
    if format == 'json':
        return (None, out if ext == '.json' else base + '.json')
    
    if format == 'both':
        # If input path has a valid extenson, use base name for both
        if ext in ('.csv', '.json'):
            base = base
        else:
            base = out
        return (base + '.csv', base + '.json')
    
    logging.error("Invalid format specified.")
    return (None, None)


def fetch_page(url: str) -> BeautifulSoup:
    """
    Fetches the content of a webpage and returns a BeautifulSoup object.
    Args:
        url (str): The URL of the webpage to fetch.
    Returns:
        BeautifulSoup: Parsed HTML content of the webpage.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        time.sleep(random.uniform(0.1, 0.5))
        return BeautifulSoup(response.content, 'lxml')
    except requests.RequestException as e:
        logging.error(f"Error fetching page {url}: {e}")
        return None
    

    
def parse_book(book_tag, base_url: str) -> dict:
    """
    Parses a book HTML tag to extract relevant information.
    Args:
        book_tag (Tag): BeautifulSoup Tag object representing a book.
        base_url (str): The base URL to resolve relative links.
    Returns:
        dict: A dictionary containing book details.
    """
    try:
        title = book_tag.h3.a['title']
        link = urljoin(base_url, book_tag.h3.a['href'])
        price_text = book_tag.find('p', class_='price_color').text
        price = float(price_text.replace('£', ''))
        star_rating = book_tag.find('p', class_='star-rating')['class'][1]
        rating = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}.get(star_rating)
        image = urljoin(base_url, book_tag.find('img')['src'])

        return {
            'Title': title,
            'Link': link,
            'Price': price,
            'Star Rating': rating,
            'Image': image
        }
    except (KeyError, ValueError, AttributeError) as e:
        logging.warning(f"Error parsing book data: {e}. Skipping this book.")
        return None
    

def parse_book_detail(soup: BeautifulSoup) -> dict:
    """
    Parses a book's detail page to extract additional information.
    """
    details = {}
    try:
        info = {
            row.find('th').text: row.find('td').text
            for row in soup.select('table.table.table-striped tr')
            if row.find('th') and row.find('td')
        }

        crumbs = soup.select('ul.breadcrumb li a')
        details['Category'] = crumbs[-1].text if len(crumbs) > 1 else 'Unknown'
        details['UPC'] = info.get('UPC', 'N/A')
        details['Product Type'] = info.get('Product Type', 'N/A')

        for key in ["Price (excl. tax)", "Price (incl. tax)", "Tax"]:
            value = info.get(key, 'N/A')
            if value:
                try:
                    details[key] = float(value.replace('£', ''))
                except ValueError:
                    details[key] = None
            else:
                details[key] = None
        
        num_reviews_str = info.get('Number of reviews', '0')
        try:
            details['Number of reviews'] = int(num_reviews_str)
        except ValueError:
            details['Number of reviews'] = 0

        availability_text = info.get('Availability', '')
        details['Availability'] = availability_text

        desc_tag = soup.select_one('#product_description ~ p')
        details['Description'] = desc_tag.text.strip() if desc_tag else 'No description available'

    except Exception as e:
        logging.warning(f"Error parsing book detail: {e}")
        return None
    
    return details


def parse_books_page(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Parses the books page to extract all book details.
    Args:
        soup (BeautifulSoup): Parsed HTML content of the books page.
        base_url (str): The base URL to resolve relative links.
    Returns:
        list[dict]: A list of dictionaries containing book details.
    """
    books = soup.find_all('article', class_='product_pod')
    book_list = []
    for book in books:
        book_data = parse_book(book, base_url)
        if book_data:
            book_list.append(book_data)
    return book_list



def update_book_with_details(book: dict) -> dict:
    """
    Fetches and updates the book dictionary with detailed information.
    Args:
        book (dict): A dictionary containing basic book details.
    Returns:
        dict: Updated dictionary with detailed book information.
    """
    logging.debug(f"Fetching details for book: {book.get('Title', 'N/A')}")
    detail_soup = fetch_page(book['Link'])
    if detail_soup:
        book_details = parse_book_detail(detail_soup)
        if book_details:
            book.update(book_details)
    return book


def save_to_file(books: list[dict], output_path: str, format: str) -> tuple[str, str]:
    """Main function to dispatch saving to CSV and/or JSON."""
    if not books:
        logging.info("No books to save.")
        return (None, None)
    
    csv_path, json_path = resolve_outputs(output_path, format)
    if csv_path:
        save_to_csv(books, csv_path)
    if json_path:   
        save_to_json(books, json_path)
    return (csv_path, json_path)


def save_to_csv(books: list[dict], output_path: str) -> None:
    """
    Saves the list of books to a CSV file.
    Args:
        books (list[dict]): List of dictionaries containing book details.
        output_path (str): The path to the output CSV file.
    """
    
    fieldnames = []
    for book in books:
        for key in book.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)



def save_to_json(books: list[dict], output_path: str) -> None:
    """
    Saves the list of books to a JSON file.
    Args:
        books (list[dict]): List of dictionaries containing book details.
        output_path (str): The path to the output JSON file.
    """
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(books, file, ensure_ascii=False, indent=4)
    logging.info(f"Data successfully saved to {output_path}")



def save_to_db(books: list[dict], db_path: str) -> None:
    """
    Saves the list of books to a SQLite database.
    Args:
        books (list[dict]): List of dictionaries containing book details.
        db_path (str): The path to the SQLite database file.
    """
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        all_keys = sorted(list(set(key for book in books for key in book.keys())))
        if not all_keys:
            logging.info("No fields to create table.")
            return
        
        sql_cols = ", ".join([f'"{key}" TEXT' for key in all_keys])

        c.execute(f"CREATE TABLE IF NOT EXISTS books ({sql_cols})")

        placeholders = ", ".join(["?" for _ in all_keys])
        insert_sql = f"INSERT INTO books VALUES ({placeholders})"

        data_to_insert = []
        for book in books:
            row = [str(book.get(key, '')) for key in all_keys]
            data_to_insert.append(row)
        
        c.executemany(insert_sql, data_to_insert)
        conn.commit()
        logging.info(f"Data successfully saved to database at {db_path}")
    
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()


def scrape_all_books(args: argparse.Namespace):
    """
    The main scraping generator function.
    It yields progress updates to the Streamlit app.

    Steps:
    1. Collect all page URLs.
    2. Parse all books from all collected pages.
    3. Fetch detailed info for each book (if details flag is set).
    4. Save results and yield final progress.
    """
    budget = PROG_WITH_DETAILS if args.details else PROG_NO_DETAILS
    yield 0

    pages_to_process = []
    current_page_url = args.url
    page_count = 0

    # Step 1: Fetch all pages and collect their URLs
    estimated_total_pages = args.limit if args.limit > 0 else 50


    while current_page_url:
        page_count += 1

        if args.limit > 0 and len(pages_to_process) >= args.limit:
            logging.info(f"Reached limit of {args.limit} pages.")
            break

        logging.info(f"Fetching page: {current_page_url}")
        soup = fetch_page(current_page_url)
        if soup is None:
            break

        pages_to_process.append((current_page_url, soup))

        initial_progress = _progress_range(page_count, estimated_total_pages, *budget['PAGES'])
        meta = {"phase": "pages", "i": page_count, "total": estimated_total_pages}
        yield (initial_progress, meta)
        
        next_button = soup.find('li', class_='next')
        if next_button and next_button.a:
            next_page_url = urljoin(current_page_url, next_button.a['href'])
            if args.delay > 0:
                time.sleep(args.delay)
            current_page_url = next_page_url
        else:
            current_page_url = None
    
    total_pages = len(pages_to_process)
    if total_pages == 0:
        logging.info("No books found to scrape.")
        yield 100
        return

    # Step 2 : Parse all books from all pages concurrently
    all_books = []
    total_pages = len(pages_to_process)
    pages_done = 0
    last_pct = budget['LISTING'][0]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(parse_books_page, soup, url): url 
                   for (url, soup) in pages_to_process}
        
        for future in as_completed(futures):
            page_url = futures[future]
            try:
                books_on_page = future.result()
                all_books.extend(books_on_page)
                pages_done += 1
                logging.info(f"Parsed page {page_url} - {len(books_on_page)} books found")
            except Exception as e:
                pages_done += 1
                logging.warning(f"Failed to parse page {pages_done}: {e}")
            
            start, end = budget['LISTING']
            progress_percent = _progress_range(pages_done, max(1, total_pages), start, end)

            progress_percent = max(progress_percent, last_pct)
            last_pct = progress_percent

            yield (progress_percent, {"phase": "listing", "i": pages_done, "total": total_pages})
    
    # Step 3: Fetch details for all books in parallel(if details flag is set)
    if args.details and all_books:
        total = len(all_books)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(update_book_with_details, book) for book in all_books]
            detailed_books = []
            
            for i, futur in enumerate(as_completed(futures), 1):
                try:
                    book = futur.result()
                    detailed_books.append(book)
                    title = (book or {}).get('Title', 'N/A')
                    logging.info(f"Fetched details for book : {title}")
                except Exception as e:
                    logging.warning(f"Detail fetch failed: {e}")
                start, end = PROG_WITH_DETAILS['DETAILS']
                progress_percent = _progress_range(i, total, start, end)

                yield (progress_percent, {"i": i, "total": total, "title": (futur.result().get('Title', 'N/A') if futur else 'N/A')})
        all_books = detailed_books
    else:
        logging.info("Skipped details (details flag is false)")

    # Step 4: Save results
    if all_books:
        csv_path, json_path = save_to_file(all_books, args.output, args.format)
        log_file_info(csv_path, "CSV ")
        log_file_info(json_path, "JSON ")
        if args.db:
            save_to_db(all_books, args.db)
            if os.path.exists(args.db):
                log_file_info(args.db, "Database ")
    yield budget['DONE']

if __name__ == '__main__':
    args = parse_args()
    scrape_all_books(args)