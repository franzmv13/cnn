import os

def count_files(folder_path):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Count the number of files
        num_files = len(files)

        return num_files

    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return None

# Example usage:
folder_path = 'data/sad'
num_files = count_files(folder_path)

if num_files is not None:
    print(f'The folder "{folder_path}" contains {num_files} files.')
