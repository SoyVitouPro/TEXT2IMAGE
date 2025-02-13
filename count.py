from collections import Counter

def count_unique_khmer_characters(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    unique_chars = set(text)
    char_counts = Counter(text)
    
    for char in unique_chars:
        print(f"{char}: {char_counts[char]}")
    
    print(f"Total unique values: {len(unique_chars)}")

# Example usage
count_unique_khmer_characters("annotation.txt")