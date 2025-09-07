def read_text_file(file_name: str) -> str:
    file = open(file_name, "r")
    system_prompt = file.read()
    file.close()
    return system_prompt
