import os
import pandas as pd
import re


def process_txt_to_csv(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = filename.replace(".txt", ".csv")
            output_file_path = os.path.join(output_folder, output_file_name)

            data = {'time': []}

            with open(input_file_path, 'r') as file:
                for line in file:
                    if line.startswith("#") or not line.strip():
                        continue

                    match = re.match(r'(\S+)\s+([\d,]+)\s+(\S+)', line.strip())
                    if match:
                        time = match.group(1)
                        value = match.group(2).replace(',', '')  
                        event = match.group(3)

                        if time not in data['time']:
                            data['time'].append(time)
                            for event_key in data.keys():
                                if event_key != 'time':
                                    data[event_key].append(0) 

                        if event not in data:
                            data[event] = [0] * (len(data['time']) - 1) 
                            data[event].append(value)
                        else:
                            data[event][-1] = value  

            df = pd.DataFrame(data)

            df.to_csv(output_file_path, index=False)
            print(f"Processed {filename} -> {output_file_path}")


input_folder = 'D:/ranote-tesla/perf/filtered'  
output_folder = 'D:/ranote-tesla/perf/csv'
process_txt_to_csv(input_folder, output_folder)
