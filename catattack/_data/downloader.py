import os

URLS = {
    'pytorch_tutorial_text': 'https://download.pytorch.org/tutorial/data.zip'
}

def _get_pytorch_tutorial_text(cache_dir='.cache', force=False):
    import urllib.request
    import zipfile

    os.makedirs(cache_dir, exist_ok=True)

    # cache_dir = os.path.relpath(cache_dir)
    zip_file_path = os.path.abspath(os.path.join(cache_dir, 'data.zip'))
    extracted_path = os.path.abspath(os.path.join(cache_dir, 'pytorch_text_data'))  # We will move the "./data" into this dir before returning

    # Greedy check if the extracted path is already there
    # Note, this does not check if the files in that path exist
    if not force and os.path.isdir(extracted_path):
        return extracted_path

    # Download the dataset
    if not os.path.isfile(zip_file_path):
        zip_file_path, _ = urllib.request.urlretrieve(URLS['pytorch_tutorial_text'], zip_file_path)
    with zipfile.ZipFile(zip_file_path, "r") as f:
        f.extractall(cache_dir)
    os.rename(os.path.join(cache_dir, 'data'), extracted_path)
    return extracted_path
