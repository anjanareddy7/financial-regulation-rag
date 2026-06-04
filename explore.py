from datasets import load_dataset

ds = load_dataset('jamescalam/ai-arxiv', split='train')
print(f'Total papers: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'Sample title: {ds[0]["title"]}')
print(f'Sample words: {len(ds[0]["content"].split())}')
print(f'Sample categories: {ds[0]["categories"]}')