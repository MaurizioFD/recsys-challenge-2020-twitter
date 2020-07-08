
import os
import sys
from TokensUtils import load_mapping, save_mapping

input_path = sys.argv[1]
output_path = sys.argv[2]

dictionary, _, _ = load_mapping(input_path)

reversed_dict = dict(map(reversed, dictionary.items()))

print("Reversed dictionary")

save_mapping(output_path, reversed_dict)

print("Compressing dictionary...")
os.system(f"gzip -k {output_path}")