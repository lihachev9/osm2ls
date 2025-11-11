ivalid_characters = "\\/:*?\"<>|"


def sanitize_filename(filename: str) -> str:
    for char in ivalid_characters:
        if char in filename:
            filename = filename.replace(char, '')
    return filename


def get_json_root_type(filename: str) -> str:
    char = "x"
    with open(filename, "r", encoding="utf-8") as f:
        # Read the file character by character
        while char != "":
            char = f.read(1)

            # Skip any whitespace
            if char.isspace():
                continue

            # If the first non-whitespace character is '{', it's a dict
            if char == "{":
                return "dict"

            # If the first non-whitespace character is '[', it's an array
            if char == "[":
                return "list"

            # If neither, the JSON file is invalid
            return "invalid"

    # If the file is empty, return "empty"
    return "empty"
