import os
import re
import sys

def update_header(filename):
    # Extract sample number from filename (e.g., 24053.03c0 -> 0)
    match = re.search(r'03(\d+)c', filename)
    if not match:
        print(f"Could not find sample number in {filename}")
        return
        return
    sample_number = match.group(1)

    # Read file contents
    with open(filename, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"{filename} is empty.")
        return

    # Modify the first line: insert sample_number before the first 'c'
    first_line = lines[0].rstrip('\n')
    c_index = first_line.find('c')
    if c_index != -1:
        new_first_line = first_line[:c_index] + sample_number + first_line[c_index:]
        lines[0] = new_first_line + '\n'
    else:
        print(f"No 'c' found in the first line of {filename}.")
        return

    # Write back to file (overwrite)
    with open(filename, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_header_ace.py <filename>")
    else:
        update_header(sys.argv[1])