import fitz
import sys

path = "2608.09687v1.pdf"
doc = fitz.open(path)
print("pages:", doc.page_count)
print("metadata:", doc.metadata)

out = []
for i, page in enumerate(doc):
    out.append(f"\n===== PAGE {i+1} =====\n")
    out.append(page.get_text())

text = "".join(out)
with open("extracted/2608_09687_extracted.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("chars:", len(text))
