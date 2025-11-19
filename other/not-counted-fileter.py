import os

def filter_invalid_timestamps_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            base_name, ext = os.path.splitext(filename)
            output_file = os.path.join(output_folder, f"{base_name}_filtered{ext}")

            invalid_timestamps = set()

            with open(input_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if '<not counted>' in line:
                        timestamp = line.split()[0]
                        invalid_timestamps.add(timestamp)

            with open(output_file, 'w') as file:
                for line in lines:
                    if line.startswith('#'):
                        file.write(line)
                        continue
                    timestamp = line.split()[0]
                    if timestamp not in invalid_timestamps:
                        file.write(line)

            print(f"save: {output_file}")

input_folder = 'D:/ranote-tesla/perf/txt' 
output_folder = 'D:/ranote-tesla/perf/filtered'  
filter_invalid_timestamps_in_folder(input_folder, output_folder)
