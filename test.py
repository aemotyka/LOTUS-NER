from data import format_validation_issues, validate_training_data


issues = validate_training_data(raise_on_error=False)

if issues:
    print("Invalid entity offsets found in train_data:\n")
    print(format_validation_issues(issues))
    raise SystemExit(1)

print("All train_data entity offsets align with spaCy token boundaries.")
