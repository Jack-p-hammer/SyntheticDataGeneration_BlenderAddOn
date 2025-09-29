from pathlib import Path

asset_library_path = Path("SyntheticDataGeneration\CustomAssets")

items_tuple = []
for item in asset_library_path.iterdir():
    items_tuple.append((item.stem, item.stem, f"Import {item.stem}"))

print(items_tuple)