import zipfile

zip_file_path = 'ch06/assets/wordsim353.zip'
extract_to_path = 'ch06/assets/'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for item in zip_ref.namelist():
        print(item)

        if item.endswith('.csv'):
            zip_ref.extract(item, extract_to_path)

#zipファイルは複数のファイルを圧縮してひとまとめにしたファイル
