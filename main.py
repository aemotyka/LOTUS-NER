import random
import spacy
from spacy.util import minibatch
from spacy.training.example import Example


train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("I need 23 chairs for the office", {"entities": [(7, 9, "QUANTITY"), (10, 16, "PRODUCT")]}),
    ("Please order 7 laptops today", {"entities": [(13, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("We bought 15 desks last week", {"entities": [(10, 12, "QUANTITY"), (13, 18, "PRODUCT")]}),
    ("Can you quote 4 monitors for me?", {"entities": [(14, 15, "QUANTITY"), (16, 24, "PRODUCT")]}),
    ("Our team needs 12 keyboards", {"entities": [(15, 17, "QUANTITY"), (18, 27, "PRODUCT")]}),
    ("She requested 9 notebooks yesterday", {"entities": [(14, 15, "QUANTITY"), (16, 25, "PRODUCT")]}),
    ("Do we have 6 printers available?", {"entities": [(11, 12, "QUANTITY"), (13, 21, "PRODUCT")]}),
    ("They delivered 18 boxes this morning", {"entities": [(15, 17, "QUANTITY"), (18, 23, "PRODUCT")]}),
    ("He wants 3 cameras for the trip", {"entities": [(9, 10, "QUANTITY"), (11, 18, "PRODUCT")]}),
    ("Add 25 pens to the supply list", {"entities": [(4, 6, "QUANTITY"), (7, 11, "PRODUCT")]}),
    ("We require 14 tables for the event", {"entities": [(11, 13, "QUANTITY"), (14, 20, "PRODUCT")]}),
    ("Could you send 8 speakers tomorrow?", {"entities": [(15, 16, "QUANTITY"), (17, 25, "PRODUCT")]}),
    ("The store sold 11 backpacks today", {"entities": [(15, 17, "QUANTITY"), (18, 27, "PRODUCT")]}),
    ("I need 5 whiteboards for training", {"entities": [(7, 8, "QUANTITY"), (9, 20, "PRODUCT")]}),
    ("Please reserve 16 microphones", {"entities": [(15, 17, "QUANTITY"), (18, 29, "PRODUCT")]}),
    ("We ordered 2 projectors last month", {"entities": [(11, 12, "QUANTITY"), (13, 23, "PRODUCT")]}),
    ("Can I get 30 folders for filing?", {"entities": [(10, 12, "QUANTITY"), (13, 20, "PRODUCT")]}),
    ("They need 13 tablets right away", {"entities": [(10, 12, "QUANTITY"), (13, 20, "PRODUCT")]}),
    ("Our office uses 20 staplers each year", {"entities": [(16, 18, "QUANTITY"), (19, 27, "PRODUCT")]}),
    ("She asked for 6 lamps in total", {"entities": [(14, 15, "QUANTITY"), (16, 21, "PRODUCT")]}),
    ("He purchased 10 shelves yesterday", {"entities": [(13, 15, "QUANTITY"), (16, 23, "PRODUCT")]}),
    ("We need 17 routers for the building", {"entities": [(8, 10, "QUANTITY"), (11, 18, "PRODUCT")]}),
    ("Please buy 21 mugs for the kitchen", {"entities": [(11, 13, "QUANTITY"), (14, 18, "PRODUCT")]}),
    ("The school ordered 24 lockers", {"entities": [(19, 21, "QUANTITY"), (22, 29, "PRODUCT")]}),
    ("Can you find 19 cables for setup?", {"entities": [(13, 15, "QUANTITY"), (16, 22, "PRODUCT")]}),
    ("I want 28 blankets for the shelter", {"entities": [(7, 9, "QUANTITY"), (10, 18, "PRODUCT")]}),
    ("They requested 22 helmets for staff", {"entities": [(15, 17, "QUANTITY"), (18, 25, "PRODUCT")]}),
    ("We need 1 scanner by Monday", {"entities": [(8, 9, "QUANTITY"), (10, 17, "PRODUCT")]}),
    ("Please ship 26 uniforms this week", {"entities": [(12, 14, "QUANTITY"), (15, 23, "PRODUCT")]}),
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

nlp.to_disk('custom_ner_model')

trained_nlp = spacy.load('custom_ner_model')

test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 6 desks?"
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f'Text: {text}')
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print()