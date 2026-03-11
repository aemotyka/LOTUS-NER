import random
import spacy
from spacy.util import minibatch
from spacy.training.example import Example


train_data = [
    ("Cartier watch with blue strap", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 13, "THING"), (19, 29, "MODIFIER")]}),
    ("Van Gogh painting of buildings and trees", {"entities": [(0, 8, "MAKER_ARTIST"), (9, 17, "THING"), (21, 40, "MODIFIER")]}),
    ("Rolex watch with black dial", {"entities": [(0, 5, "MAKER_ARTIST"), (6, 11, "THING"), (17, 27, "MODIFIER")]}),
    ("Tiffany necklace with diamonds", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 16, "THING"), (22, 30, "MODIFIER")]}),
    ("Picasso drawing of a woman", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 15, "THING"), (19, 26, "MODIFIER")]}),
    ("Monet painting of water lilies", {"entities": [(0, 5, "MAKER_ARTIST"), (6, 14, "THING"), (18, 30, "MODIFIER")]}),
    ("Cartier bracelet in gold", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 17, "THING"), (21, 25, "MODIFIER")]}),
    ("Bulgari ring with sapphire", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 12, "THING"), (18, 26, "MODIFIER")]}),
    ("Van Cleef bracelet with onyx", {"entities": [(0, 9, "MAKER_ARTIST"), (10, 19, "THING"), (25, 30, "MODIFIER")]}),
    ("Patek Philippe watch in platinum", {"entities": [(0, 14, "MAKER_ARTIST"), (15, 20, "THING"), (24, 32, "MODIFIER")]}),
    ("Ancient Roman tiles with geometric patterns", {"entities": [(0, 13, "MAKER_ARTIST"), (14, 19, "THING"), (25, 43, "MODIFIER")]}),
    ("Chinese vase with dragon motif", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 12, "THING"), (18, 30, "MODIFIER")]}),
    ("French cabinet in mahogany", {"entities": [(0, 6, "MAKER_ARTIST"), (7, 14, "THING"), (18, 27, "MODIFIER")]}),
    ("Italian mirror with gilt frame", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 14, "THING"), (20, 30, "MODIFIER")]}),
    ("Persian rug with floral design", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 11, "THING"), (17, 30, "MODIFIER")]}),
    ("Louis Vuitton trunk with brass corners", {"entities": [(0, 13, "MAKER_ARTIST"), (14, 19, "THING"), (25, 38, "MODIFIER")]}),
    ("Hermes bag in red leather", {"entities": [(0, 6, "MAKER_ARTIST"), (7, 10, "THING"), (14, 24, "MODIFIER")]}),
    ("Faberge box with enamel flowers", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 11, "THING"), (17, 31, "MODIFIER")]}),
    ("Ming bowl with blue decoration", {"entities": [(0, 4, "MAKER_ARTIST"), (5, 9, "THING"), (15, 30, "MODIFIER")]}),
    ("Roman bust in marble", {"entities": [(0, 5, "MAKER_ARTIST"), (6, 10, "THING"), (14, 20, "MODIFIER")]}),
    ("Cartier brooch with rubies and diamonds", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 14, "THING"), (20, 40, "MODIFIER")]}),
    ("Van Gogh drawing with village scene", {"entities": [(0, 8, "MAKER_ARTIST"), (9, 16, "THING"), (22, 35, "MODIFIER")]}),
    ("Rolex watch with green bezel", {"entities": [(0, 5, "MAKER_ARTIST"), (6, 11, "THING"), (17, 28, "MODIFIER")]}),
    ("Tiffany bracelet with heart charms", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 17, "THING"), (23, 35, "MODIFIER")]}),
    ("Picasso painting of musicians", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 16, "THING"), (20, 29, "MODIFIER")]}),
    ("Monet watercolor of gardens", {"entities": [(0, 5, "MAKER_ARTIST"), (6, 16, "THING"), (20, 27, "MODIFIER")]}),
    ("Bulgari necklace with emerald pendant", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 16, "THING"), (22, 37, "MODIFIER")]}),
    ("Patek Philippe clock with moonphase", {"entities": [(0, 14, "MAKER_ARTIST"), (15, 20, "THING"), (26, 35, "MODIFIER")]}),
    ("Chinese screen with lacquer panels", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 14, "THING"), (20, 34, "MODIFIER")]}),
    ("French chair with carved arms", {"entities": [(0, 6, "MAKER_ARTIST"), (7, 12, "THING"), (18, 29, "MODIFIER")]}),
    ("Italian table in walnut", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 13, "THING"), (17, 23, "MODIFIER")]}),
    ("Persian carpet with medallion pattern", {"entities": [(0, 7, "MAKER_ARTIST"), (8, 14, "THING"), (20, 38, "MODIFIER")]}),
]


nlp = spacy.load('en_core_web_lg')

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for _, annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f'Epoch {epoch + 1}, Losses: {losses}')

nlp.to_disk('custom_query_derivation')

trained_nlp = spacy.load('custom_query_derivation')

test_texts = [
    "Ancient Egyptian Stela",
    "Russian lockbox with Art Deco key",
    "Arabian camel saddle made of leather"
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f'Text: {text}')
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print()