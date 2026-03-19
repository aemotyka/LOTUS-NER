import spacy

from data.train import train_data


def find_invalid_entity_offsets(dataset=None):
    dataset = train_data if dataset is None else dataset
    nlp = spacy.blank("en")
    issues = []

    for example_index, (text, annotations) in enumerate(dataset, start=1):
        doc = nlp.make_doc(text)
        tokens = [f"{token.text}[{token.idx}:{token.idx + len(token)}]" for token in doc]
        entities = list(annotations.get("entities", []))

        for entity_index, (start, end, label) in enumerate(entities, start=1):
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

        sorted_entities = sorted(
            enumerate(entities, start=1),
            key=lambda item: (item[1][0], item[1][1], item[1][2]),
        )
        for sorted_index, (entity_index, (start, end, label)) in enumerate(sorted_entities):
            for other_entity_index, (other_start, other_end, other_label) in sorted_entities[sorted_index + 1:]:
                if other_start >= end:
                    break

                if max(start, other_start) < min(end, other_end):
                    issues.append({
                        "example_index": example_index,
                        "entity_index": entity_index,
                        "text": text,
                        "start": start,
                        "end": end,
                        "label": label,
                        "slice": text[start:end],
                        "reason": "entity overlaps with another entity",
                        "tokens": tokens,
                        "other_entity_index": other_entity_index,
                        "other_start": other_start,
                        "other_end": other_end,
                        "other_label": other_label,
                        "other_slice": text[other_start:other_end],
                    })

    return issues


def format_validation_issues(issues):
    sections = []
    for issue in issues:
        lines = [
            f'Example {issue["example_index"]}: "{issue["text"]}"',
            (
                f'Entity {issue["entity_index"]}: '
                f'({issue["start"]}, {issue["end"]}, "{issue["label"]}")'
            ),
            f'Reason: {issue["reason"]}',
            f'Slice: {issue["slice"]!r}',
        ]

        if "other_entity_index" in issue:
            lines.extend([
                (
                    f'Overlaps With Entity {issue["other_entity_index"]}: '
                    f'({issue["other_start"]}, {issue["other_end"]}, "{issue["other_label"]}")'
                ),
                f'Other Slice: {issue["other_slice"]!r}',
            ])

        lines.append(f'Tokens: {", ".join(issue["tokens"])}')
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def validate_training_data(dataset=None, raise_on_error=True):
    issues = find_invalid_entity_offsets(dataset=dataset)
    if issues and raise_on_error:
        raise ValueError(
            "Invalid entity offsets found in training data.\n\n"
            f"{format_validation_issues(issues)}"
        )
    return issues
