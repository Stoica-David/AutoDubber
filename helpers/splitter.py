with open("../examples/transcript.txt", "r") as file:
    content = file.read()

for phrase in content.split("."):
    print(f"len = {len(phrase)} :{phrase}\n")
