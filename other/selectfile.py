import os
import shutil


txt_file_path = 'C:/Users/Xi/Desktop/ransom/sodinokibi/selected.txt' 
source_folder = 'C:/Users/Xi/Desktop/ransom/sodinokibi/' 
selected_folder = 'C:/Users/Xi/Desktop/ransom/sodinokibi/selected' 

if not os.path.exists(selected_folder):
    os.makedirs(selected_folder)

with open(txt_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            file_name = parts[3]

            if file_name.endswith('.exe'):
                zip_file_name = file_name[:-4] + '.zip'  
                full_zip_file_path = os.path.join(source_folder, zip_file_name)  

                if os.path.exists(full_zip_file_path):
                    shutil.move(full_zip_file_path, os.path.join(selected_folder, zip_file_name))
                    print(f"文件 {zip_file_name} 2 {selected_folder} ")
                else:
                    print(f"{zip_file_name} not in {source_folder}")

print("OK!")

