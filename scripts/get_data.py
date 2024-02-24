
from pathlib import Path
import requests
import zipfile
import os

def download_data():
    dataset_dir = Path('./data/raw')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    notEmpty = any(dataset_dir.iterdir())
    
    if notEmpty:
        print('Dataset exists.')
        
    else:
        try:
            response = requests.get('https://storage.googleapis.com/kaggle-data-sets/3697155/6410731/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240224T212811Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3630dc54d8e2cee4459eceb6d3414ccb669f04b6996660b9d1a5e20d07f242fde686cf4609e222e2e0d4d34746a77c1c0115c550228a80bfb707e252614ae108f6e2b7f6fa206998100df0c3218b91bd5ad6ea64aa2921b4ecb170f123e0e9e36e9e20a0d772e1689d698fa53a1f1f0f673cc4b94b42919f970c6286bd3d2fa7ecf5e72a14a3c4ba8fd32e2074c97e178e922d8a44280914e36b8371ebc172e122d9db33e6bd83735ba3c3f106224e2eb6566d7885fd87dccd26156f7018ec0d1d4138b55b4d27ba205e5fd68e4b923b4ca8b64bced817e37f9164e3284bab015e05ec046bf635f90f18ebf1fcfcc2ab450851c441deea8700d717f33251be3a')
        
            if response.status_code == 200:
                print('Downloading dataset..')
                with open('archive.zip', 'wb') as f:
                    f.write(response.content)
        
                print('Unzipping...')
                with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                    print('Done.')
        
                os.remove('archive.zip')
    
            else:
                raise requests.exceptions.RequestException(f"Error downloading dataset. status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(e)
