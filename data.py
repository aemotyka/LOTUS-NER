import spacy


train_data = [('Van Gogh painting', {'entities': [(0, 8, 'MAKER_ARTIST'), (9, 17, 'TYPE')]}),
 ('Drawing by Renoir', {'entities': [(0, 7, 'TYPE'), (11, 17, 'MAKER_ARTIST')]}),
 ('Charcoal sketch by Monet',
  {'entities': [(0, 8, 'DESCRIPTOR'), (9, 15, 'TYPE'), (19, 24, 'MAKER_ARTIST')]}),
 ('Tiffany & Co. diamond earrings with gold',
  {'entities': [(0, 13, 'MAKER_ARTIST'),
                (14, 21, 'DESCRIPTOR'),
                (22, 30, 'TYPE'),
                (36, 40, 'DESCRIPTOR')]}),
 ('Art Deco bracelet by Tom Ford',
  {'entities': [(0, 8, 'DESCRIPTOR'), (9, 17, 'TYPE'), (21, 29, 'MAKER_ARTIST')]}),
 ('Royal blue', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 10, 'DESCRIPTOR')]}),
 ('Ancient vase', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 12, 'TYPE')]}),
 ('Chinese jade tiles',
  {'entities': [(0, 7, 'DESCRIPTOR'), (8, 12, 'DESCRIPTOR'), (13, 18, 'TYPE')]}),
 ('WWII merit medal',
  {'entities': [(0, 4, 'DESCRIPTOR'), (5, 10, 'DESCRIPTOR'), (11, 16, 'TYPE')]}),
 ('Cityscape sketch', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 16, 'TYPE')]}),
 ('Enigma machine', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 14, 'TYPE')]}),
 ('French perfume bottle',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 14, 'DESCRIPTOR'), (15, 21, 'TYPE')]}),
 ('Bottle from France', {'entities': [(0, 6, 'TYPE'), (12, 18, 'DESCRIPTOR')]}),
 ('Flora Danica dining set', {'entities': [(0, 12, 'DESCRIPTOR'), (13, 23, 'TYPE')]}),
 ('German porcelain plates',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 16, 'DESCRIPTOR'), (17, 23, 'TYPE')]}),
 ('Silver coffee pot by Paul Revere',
  {'entities': [(0, 6, 'DESCRIPTOR'),
                (7, 13, 'DESCRIPTOR'),
                (14, 17, 'TYPE'),
                (21, 32, 'MAKER_ARTIST')]}),
 ('Paul Revere', {'entities': [(0, 11, 'MAKER_ARTIST')]}),
 ('Ancient Chinese', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 15, 'DESCRIPTOR')]}),
 ('Argentinian meteorite', {'entities': [(0, 11, 'DESCRIPTOR'), (12, 21, 'TYPE')]}),
 ('Cartier watch with blue strap',
  {'entities': [(0, 7, 'MAKER_ARTIST'),
                (8, 13, 'TYPE'),
                (19, 23, 'DESCRIPTOR'),
                (24, 29, 'DESCRIPTOR')]}),
 ('Rolex wristwatch', {'entities': [(0, 5, 'MAKER_ARTIST'), (6, 16, 'TYPE')]}),
 ('Patek Philippe pocket watch',
  {'entities': [(0, 14, 'MAKER_ARTIST'), (15, 21, 'DESCRIPTOR'), (22, 27, 'TYPE')]}),
 ('Omega steel watch',
  {'entities': [(0, 5, 'MAKER_ARTIST'), (6, 11, 'DESCRIPTOR'), (12, 17, 'TYPE')]}),
 ('Cartier gold ring',
  {'entities': [(0, 7, 'MAKER_ARTIST'), (8, 12, 'DESCRIPTOR'), (13, 17, 'TYPE')]}),
 ('Sapphire ring by Bulgari',
  {'entities': [(0, 8, 'DESCRIPTOR'), (9, 13, 'TYPE'), (17, 24, 'MAKER_ARTIST')]}),
 ('Ruby necklace by Cartier',
  {'entities': [(0, 4, 'DESCRIPTOR'), (5, 13, 'TYPE'), (17, 24, 'MAKER_ARTIST')]}),
 ('Emerald bracelet', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 16, 'TYPE')]}),
 ('Diamond brooch by Tiffany',
  {'entities': [(0, 7, 'DESCRIPTOR'), (8, 14, 'TYPE'), (18, 25, 'MAKER_ARTIST')]}),
 ('Platinum pendant', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 16, 'TYPE')]}),
 ('Art Nouveau pendant by Lalique',
  {'entities': [(0, 11, 'DESCRIPTOR'), (12, 19, 'TYPE'), (23, 30, 'MAKER_ARTIST')]}),
 ('Oil painting by Picasso',
  {'entities': [(0, 3, 'DESCRIPTOR'), (4, 12, 'TYPE'), (16, 23, 'MAKER_ARTIST')]}),
 ('Watercolor painting by Sargent',
  {'entities': [(0, 10, 'DESCRIPTOR'), (11, 19, 'TYPE'), (23, 30, 'MAKER_ARTIST')]}),
 ('Landscape painting', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 18, 'TYPE')]}),
 ('Portrait painting by Degas',
  {'entities': [(0, 8, 'DESCRIPTOR'), (9, 17, 'TYPE'), (21, 26, 'MAKER_ARTIST')]}),
 ('Portrait miniature', {'entities': [(0, 8, 'TYPE'), (9, 18, 'DESCRIPTOR')]}),
 ('Charcoal drawing', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 16, 'TYPE')]}),
 ('Ink drawing by Rembrandt',
  {'entities': [(0, 3, 'DESCRIPTOR'), (4, 11, 'TYPE'), (15, 24, 'MAKER_ARTIST')]}),
 ('Bronze sculpture by Rodin',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 16, 'TYPE'), (20, 25, 'MAKER_ARTIST')]}),
 ('Marble sculpture', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 16, 'TYPE')]}),
 ('Pastel sketch of horses',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 13, 'TYPE'), (17, 23, 'DESCRIPTOR')]}),
 ('Mahogany chair', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 14, 'TYPE')]}),
 ('Walnut table', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 12, 'TYPE')]}),
 ('Coffee table', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 12, 'TYPE')]}),
 ('Dining table by Stickley',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 12, 'TYPE'), (16, 24, 'MAKER_ARTIST')]}),
 ('Oak cabinet', {'entities': [(0, 3, 'DESCRIPTOR'), (4, 11, 'TYPE')]}),
 ('Victorian cabinet', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 17, 'TYPE')]}),
 ('Art Deco chair', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 14, 'TYPE')]}),
 ('Louis XV armchair', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 17, 'TYPE')]}),
 ('Pair of chairs', {'entities': [(0, 4, 'DESCRIPTOR'), (8, 14, 'TYPE')]}),
 ('Leather sofa', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 12, 'TYPE')]}),
 ('Chinese vase', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 12, 'TYPE')]}),
 ('Ming vase', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 9, 'TYPE')]}),
 ('Porcelain vase by Meissen',
  {'entities': [(0, 9, 'DESCRIPTOR'), (10, 14, 'TYPE'), (18, 25, 'MAKER_ARTIST')]}),
 ('Jade snuff bottle',
  {'entities': [(0, 4, 'DESCRIPTOR'), (5, 10, 'DESCRIPTOR'), (11, 17, 'TYPE')]}),
 ('Cloisonne bowl', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 14, 'TYPE')]}),
 ('Bronze censer', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 13, 'TYPE')]}),
 ('Ceramic jar', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 11, 'TYPE')]}),
 ('Glass vase', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 10, 'TYPE')]}),
 ('Crystal decanter', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 16, 'TYPE')]}),
 ('Stoneware pitcher', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 17, 'TYPE')]}),
 ('Ancient Roman coin', {'entities': [(0, 13, 'DESCRIPTOR'), (14, 18, 'TYPE')]}),
 ('Gold coin', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 9, 'TYPE')]}),
 ('Silver medal', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 12, 'TYPE')]}),
 ('Military badge', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 14, 'TYPE')]}),
 ('Civil War sword', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 15, 'TYPE')]}),
 ('WWI helmet', {'entities': [(0, 3, 'DESCRIPTOR'), (4, 10, 'TYPE')]}),
 ('Navy telescope', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 14, 'TYPE')]}),
 ('Brass compass', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 13, 'TYPE')]}),
 ('Marine chronometer', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 18, 'TYPE')]}),
 ('Aviation map', {'entities': [(0, 8, 'DESCRIPTOR'), (9, 12, 'TYPE')]}),
 ('Paraiba sapphire necklace',
  {'entities': [(0, 7, 'DESCRIPTOR'), (8, 16, 'DESCRIPTOR'), (17, 25, 'TYPE')]}),
 ('Blue sapphire ring',
  {'entities': [(0, 4, 'DESCRIPTOR'), (5, 13, 'DESCRIPTOR'), (14, 18, 'TYPE')]}),
 ('Yellow diamond bracelet',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 14, 'DESCRIPTOR'), (15, 23, 'TYPE')]}),
 ('Black onyx earrings',
  {'entities': [(0, 5, 'DESCRIPTOR'), (6, 10, 'DESCRIPTOR'), (11, 19, 'TYPE')]}),
 ('Red coral necklace',
  {'entities': [(0, 3, 'DESCRIPTOR'), (4, 9, 'DESCRIPTOR'), (10, 18, 'TYPE')]}),
 ('Pearl brooch', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 12, 'TYPE')]}),
 ('Opal pendant', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 12, 'TYPE')]}),
 ('Turquoise cuff', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 14, 'TYPE')]}),
 ('Garnet ring by David Yurman',
  {'entities': [(0, 6, 'DESCRIPTOR'), (7, 11, 'TYPE'), (15, 27, 'MAKER_ARTIST')]}),
 ('Onyx gold bracelet by Van Cleef & Arpels',
  {'entities': [(0, 4, 'DESCRIPTOR'),
                (5, 9, 'DESCRIPTOR'),
                (10, 18, 'TYPE'),
                (22, 40, 'MAKER_ARTIST')]}),
 ('Modernist lamp', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 14, 'TYPE')]}),
 ('Art Deco lamp by Tiffany Studios',
  {'entities': [(0, 8, 'DESCRIPTOR'), (9, 13, 'TYPE'), (17, 32, 'MAKER_ARTIST')]}),
 ('Brass chandelier', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 16, 'TYPE')]}),
 ('Glass lantern', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 13, 'TYPE')]}),
 ('Silver candelabra', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 17, 'TYPE')]}),
 ('Crystal sconce', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 14, 'TYPE')]}),
 ('Industrial floor lamp',
  {'entities': [(0, 10, 'DESCRIPTOR'), (11, 16, 'DESCRIPTOR'), (17, 21, 'TYPE')]}),
 ('Bronze torchere', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 15, 'TYPE')]}),
 ('Ceramic table lamp',
  {'entities': [(0, 7, 'DESCRIPTOR'), (8, 13, 'DESCRIPTOR'), (14, 18, 'TYPE')]}),
 ('French mirror', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 13, 'TYPE')]}),
 ('Persian rug', {'entities': [(0, 7, 'DESCRIPTOR'), (8, 11, 'TYPE')]}),
 ('Silk carpet', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 11, 'TYPE')]}),
 ('Wool tapestry', {'entities': [(0, 4, 'DESCRIPTOR'), (5, 13, 'TYPE')]}),
 ('Floral tapestry', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 15, 'TYPE')]}),
 ('Geometric carpet', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 16, 'TYPE')]}),
 ('Needlepoint pillow', {'entities': [(0, 11, 'DESCRIPTOR'), (12, 18, 'TYPE')]}),
 ('Embroidered textile', {'entities': [(0, 11, 'DESCRIPTOR'), (12, 19, 'TYPE')]}),
 ('Linen sampler', {'entities': [(0, 5, 'DESCRIPTOR'), (6, 13, 'TYPE')]}),
 ('Patchwork quilt', {'entities': [(0, 9, 'DESCRIPTOR'), (10, 15, 'TYPE')]}),
 ('Velvet screen', {'entities': [(0, 6, 'DESCRIPTOR'), (7, 13, 'TYPE')]}),
 ('Baccarat crystal vase',
  {'entities': [(0, 8, 'MAKER_ARTIST'), (9, 16, 'DESCRIPTOR'), (17, 21, 'TYPE')]}),
 ('Wedgwood jasperware vase',
  {'entities': [(0, 8, 'MAKER_ARTIST'), (9, 19, 'DESCRIPTOR'), (20, 24, 'TYPE')]}),
 ('Steuben glass bowl',
  {'entities': [(0, 7, 'MAKER_ARTIST'), (8, 13, 'DESCRIPTOR'), (14, 18, 'TYPE')]}),
 ('Hermes silk scarf',
  {'entities': [(0, 6, 'MAKER_ARTIST'), (7, 11, 'DESCRIPTOR'), (12, 17, 'TYPE')]}),
 ('Louis Vuitton leather trunk',
  {'entities': [(0, 13, 'MAKER_ARTIST'), (14, 21, 'DESCRIPTOR'), (22, 27, 'TYPE')]}),
 ('Gucci bamboo bag',
  {'entities': [(0, 5, 'MAKER_ARTIST'), (6, 12, 'DESCRIPTOR'), (13, 16, 'TYPE')]}),
 ('Prada nylon bag',
  {'entities': [(0, 5, 'MAKER_ARTIST'), (6, 11, 'DESCRIPTOR'), (12, 15, 'TYPE')]}),
 ('Burberry trench coat',
  {'entities': [(0, 8, 'MAKER_ARTIST'), (9, 15, 'DESCRIPTOR'), (16, 20, 'TYPE')]}),
 ('Dior perfume bottle',
  {'entities': [(0, 4, 'MAKER_ARTIST'), (5, 12, 'DESCRIPTOR'), (13, 19, 'TYPE')]}),
 ('Chanel brooch with pearls',
  {'entities': [(0, 6, 'MAKER_ARTIST'), (7, 13, 'TYPE'), (19, 25, 'DESCRIPTOR')]})]

test_data = [
    "Cartier watch with blue strap",
    "Van Gogh painting of buildings and trees",
    "Rolex watch with black dial",
    "Tiffany necklace with diamonds",
    "Picasso drawing of a woman",
    "Monet painting of water lilies",
    "Cartier bracelet in gold",
    "Bulgari ring with sapphire",
    "Van Cleef bracelet with onyx",
    "Patek Philippe watch in platinum",
    "Ancient Roman tiles with geometric patterns",
    "Chinese vase with dragon motif",
    "French cabinet in mahogany",
    "Italian mirror with gilt frame",
    "Persian rug with floral design",
    "Louis Vuitton trunk with brass corners",
    "Hermes bag in red leather",
    "Faberge box with enamel flowers",
    "Ming bowl with blue decoration",
    "Roman bust in marble",
    "Cartier brooch with rubies and diamonds",
    "Van Gogh drawing with village scene",
    "Rolex watch with green bezel",
    "Tiffany bracelet with heart charms",
    "Picasso painting of musicians",
    "Monet watercolor of gardens",
    "Bulgari necklace with emerald pendant",
    "Patek Philippe clock with moonphase",
    "Chinese screen with lacquer panels",
    "French chair with carved arms",
    "Italian table in walnut",
    "Persian carpet with medallion pattern"
]


def find_invalid_entity_offsets(dataset=None):
    dataset = train_data if dataset is None else dataset
    nlp = spacy.blank("en")
    issues = []

    for example_index, (text, annotations) in enumerate(dataset, start=1):
        doc = nlp.make_doc(text)
        tokens = [f"{token.text}[{token.idx}:{token.idx + len(token)}]" for token in doc]

        for entity_index, (start, end, label) in enumerate(annotations.get("entities", []), start=1):
            snippet = text[max(0, start):min(len(text), end)]

            if start < 0 or end > len(text) or start >= end:
                issues.append({
                    "example_index": example_index,
                    "entity_index": entity_index,
                    "text": text,
                    "start": start,
                    "end": end,
                    "label": label,
                    "slice": snippet,
                    "reason": "offsets are out of range for the text",
                    "tokens": tokens,
                })
                continue

            if doc.char_span(start, end, label=label, alignment_mode="strict") is None:
                issues.append({
                    "example_index": example_index,
                    "entity_index": entity_index,
                    "text": text,
                    "start": start,
                    "end": end,
                    "label": label,
                    "slice": snippet,
                    "reason": "offsets do not align to spaCy token boundaries",
                    "tokens": tokens,
                })

    return issues


def format_validation_issues(issues):
    sections = []
    for issue in issues:
        sections.append(
            "\n".join([
                f'Example {issue["example_index"]}: "{issue["text"]}"',
                (
                    f'Entity {issue["entity_index"]}: '
                    f'({issue["start"]}, {issue["end"]}, "{issue["label"]}")'
                ),
                f'Reason: {issue["reason"]}',
                f'Slice: {issue["slice"]!r}',
                f'Tokens: {", ".join(issue["tokens"])}',
            ])
        )
    return "\n\n".join(sections)


def validate_training_data(dataset=None, raise_on_error=True):
    issues = find_invalid_entity_offsets(dataset=dataset)
    if issues and raise_on_error:
        raise ValueError(
            "Invalid entity offsets found in training data.\n\n"
            f"{format_validation_issues(issues)}"
        )
    return issues
