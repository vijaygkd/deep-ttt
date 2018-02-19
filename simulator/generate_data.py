from simulator import simulate_all_boards

data = simulate_all_boards()

file_path = "data.csv"
print("Writing data to file:", file_path)
with open(file_path, mode='w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(data))
