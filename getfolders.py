from pathlib import Path

def get_formatted_folders(target_directory="."):
    """
    Scans the target directory and returns folder names 
    wrapped in quotes and separated by commas.
    """
    path = Path("C:\\Users\\succe\\Desktop\\Projects\\Capstone Project Group 6 C6 AIML Track A\\plantvillage-dataset\\raw\\color")
    
    if not path.exists():
        return f"Error: The path '{target_directory}' does not exist."

    # Grab the names of all directories
    folder_names = [item.name for item in path.iterdir() if item.is_dir()]
    
    # Wrap each name in quotes and join them with a comma and a space
    result = ", ".join([f"'{name}'" for name in folder_names])
    
    return result

if __name__ == "__main__":
    # Point this to your raw/color directory
    target_path = "." 
    
    folders = get_formatted_folders(target_path)
    
    print("\n--- Copy and Paste This ---")
    print(folders)
    print("---------------------------\n")