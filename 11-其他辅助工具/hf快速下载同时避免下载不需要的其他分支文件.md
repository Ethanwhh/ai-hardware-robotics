快速下载download\_hf\_files.py

python3 download\_hf\_files.py IPEC-COMMUNITY/spatialvla-4b-224-pt main --repo-type model --download\_path /data1/spatialvla-4b-224-pt

python3 download\_hf\_files.py google/gemma-3-1b-it main --repo-type model --download\_path ./gemma-3-1b-it

python3 download\_hf\_files.py lerobot/smolvla_base main --repo-type model --download\_path ./smolvla_base

python3 download\_hf\_files.py nikriz/aopoli-lv-libero_combined_no_noops_lerobot_v21 main --repo-type dataset --download\_path /home/vipuser/217data/aopoli-lv-libero

python3 hf_downloader.py nikriz/aopoli-lv-libero_combined_no_noops_lerobot_v21 main --repo-type dataset --download\_path /home/vipuser/217data/aopoli-lv-libero-new

python3 hf_downloader.py openvla/modified_libero_rlds main --repo-type dataset --download_path ./modified_libero_rlds_data

```python
import os
import requests
import json
import argparse
from urllib.parse import urljoin, quote

# <--- MODIFIED! --- 添加了 hf_token 参数
def fetch_file_list(repo_id, branch, repo_type, hf_token):
    """
    Fetches a recursive list of files from a Hugging Face repository.
    repo_type can be 'model' or 'dataset'.
    """
    api_url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}?recursive=true"
    print(f"Fetching file list from: {api_url}")
  
    # <--- MODIFIED! --- 添加 Authorization header 以进行身份验证
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
  
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch data from {api_url}. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        print("Please ensure your HF_TOKEN is correct and has the necessary permissions.")
        exit(1)
    return response.json()

# <--- MODIFIED! --- 添加了 hf_token 参数
def download_files(file_list, repo_id, branch, download_path, repo_type, hf_token):
    """
    Downloads files using aria2c, preserving the original directory structure.
    """
    os.makedirs(download_path, exist_ok=True)
    print(f"Ensuring root download directory exists: {download_path}")

    download_links_with_out = []
  
    if repo_type == 'dataset':
        base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/"
    else:
        base_url = f"https://huggingface.co/{repo_id}/resolve/{branch}/"

    for file_info in file_list:
        if file_info.get('type') == 'file':
            file_path = file_info['path']
  
            local_file_dir = os.path.dirname(os.path.join(download_path, file_path))
            if local_file_dir:
                os.makedirs(local_file_dir, exist_ok=True)
  
            file_url = urljoin(base_url, quote(file_path, safe=''))
            aria2c_options = f"  out={file_path}"
  
            download_links_with_out.append((file_url, aria2c_options))

    if not download_links_with_out:
        print("No files found to download.")
        return

    download_file = os.path.join(download_path, "download_links.txt")
    with open(download_file, "w") as f:
        for url, options in download_links_with_out:
            f.write(f"{url}\n{options}\n")

    print(f"Found {len(download_links_with_out)} files. Links and output paths saved to {download_file}")
  
    # <--- MODIFIED! --- 为 aria2c 命令添加 header 选项
    aria2_header_option = ""
    if hf_token:
        # 注意这里的引号处理，确保 shell 能正确解析
        aria2_header_option = f'--header="Authorization: Bearer {hf_token}"'

    print("Starting download with aria2c (preserving directory structure)...")
    # 将 header 选项加入命令中
    command = f'aria2c -x 16 -c --retry-wait=5 --max-tries=0 {aria2_header_option} -i "{download_file}" -d "{download_path}"'
    os.system(command)


def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face repository using aria2c")
    parser.add_argument("repo_id", help="The Hugging Face repository ID (e.g., 'google/gemma-3-1b-it')")
    parser.add_argument("branch", help="The branch to download from (e.g., 'main')")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"], help="Type of the repository ('model' or 'dataset'). Default is 'model'.")
    parser.add_argument("--download_path", default=None, help="The path to save the downloaded files. Defaults to a directory named after the repo_id.")
  
    args = parser.parse_args()
  
    # <--- MODIFIED! --- 从环境变量中读取 Hugging Face Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable is not set. Downloads for gated models may fail.")
        print("You can get a token from https://huggingface.co/settings/tokens")

    download_path = args.download_path
    if download_path is None:
        download_path = f"./{args.repo_id.split('/')[-1]}"

    # <--- MODIFIED! --- 传递 token
    file_list = fetch_file_list(args.repo_id, args.branch, args.repo_type, hf_token)
  
    # <--- MODIFIED! --- 传递 token
    download_files(file_list, args.repo_id, args.branch, download_path, args.repo_type, hf_token)
  
    print(f"Download completed! Files are saved to {download_path}")

if __name__ == "__main__":
    main()
```
