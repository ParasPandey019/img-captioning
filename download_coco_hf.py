import os
import time
import requests
from tqdm import tqdm

def download_resumable(url, dst_path, max_retries=100):
    retries = 0
    while retries < max_retries:
        try:
            resume_byte_pos = os.path.getsize(dst_path) if os.path.exists(dst_path) else 0
            headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos else {}
            response = requests.get(url, headers=headers, stream=True, timeout=10)
            total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos

            mode = "ab" if resume_byte_pos else "wb"
            with open(dst_path, mode) as f, tqdm(
                initial=resume_byte_pos,
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=os.path.basename(dst_path)
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return  # Download succeeded

        except (requests.ConnectionError, requests.Timeout) as e:
            retries += 1
            print(f"\nNetwork error ({e}), retrying {retries}/{max_retries} in 5 min...")
            time.sleep(300)  # <-- wait 5 minutes between retries

    raise Exception(f"Failed to download {url} after {max_retries} retries.")

target_folder = "./coco2017"
os.makedirs(target_folder, exist_ok=True)

files = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

for fname, url in files.items():
    dst = os.path.join(target_folder, fname)
    print(f"\n↓ Downloading {fname}")
    download_resumable(url, dst)

print("All files completed.")
