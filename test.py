for text, ann in train_data:
    for start, end, label in ann["entities"]:
        print(repr(text[start:end]), label)