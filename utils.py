import pdb
import os
import dotenv
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Set
import time
import json

PERSIST_DIRECTORY = "content"

def get_all_urls_and_images(base_url: str) -> tuple[Set[str], Set[str]]:
    urls_found, image_urls = set(), []
    url_set = set()
    urls_to_scan, scanned_urls = {base_url}, set()
    parsed_base = urlparse(base_url)
    base_domain, base_path = parsed_base.netloc, parsed_base.path.rstrip('/')
    image_dir = os.path.join(PERSIST_DIRECTORY, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    while urls_to_scan:
        current_url = urls_to_scan.pop()
        if current_url in scanned_urls:
            continue
            
        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            scanned_urls.add(current_url)
            
            if current_url.startswith(base_url):
                urls_found.add(current_url)
                for img in soup.find_all('img'):
                    img_url = img.get('src')
                    if img_url:
                        img_url = urljoin(current_url, img_url)
                        if img_url not in url_set:
                            url_set.add(img_url)
                            image_urls.append({"url": img_url, "source": current_url})
                            # try:
                            #     img_response = requests.get(img_url, timeout=10)
                            #     if img_response.status_code == 200:
                            #         img_name = os.path.basename(urlparse(img_url).path)
                            #         if img_name:
                            #             with open(os.path.join(image_dir, img_name), 'wb') as f:
                            #                 f.write(img_response.content)
                            # except Exception:
                            #     continue
       
            for link in soup.find_all(['a']):
                href = link.get('href')
                if not href:
                    continue
                
                url = urljoin(current_url, href)
                parsed_url = urlparse(url)
                
                if (parsed_url.scheme in ['http', 'https'] and
                    parsed_url.netloc == base_domain and
                    parsed_url.path.startswith(base_path) and
                    not any(url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip'])):
                    clean_url = url.split('#')[0].rstrip('/')
                    if (clean_url.startswith(base_url) and
                        clean_url not in scanned_urls and 
                        clean_url not in urls_to_scan):
                        urls_to_scan.add(clean_url)
            
            time.sleep(1)
            
        except Exception:
            continue
    
    return urls_found, image_urls

if __name__ == "__main__":
    base_url = "https://science.nasa.gov/mars/"
    urls, images = get_all_urls_and_images(base_url)
    with open("images.json", "w") as file:
        json.dump(images, file)

    