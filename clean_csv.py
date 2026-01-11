import csv

INPUT_FILE = "dataset/apple_support.csv"
OUTPUT_FILE = "dataset/apple_support_clean.csv"

cleaned_rows = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header

    cleaned_rows.append(["question", "answer"])

    for row in reader:
        if len(row) >= 2:
            question = row[0].strip()
            answer = ",".join(row[1:]).strip()
            cleaned_rows.append([question, answer])

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(cleaned_rows)

print("✅ CSV berhasil dibersihkan!")
print(f"➡ File baru: {OUTPUT_FILE}")
