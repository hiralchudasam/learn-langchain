# Assignment 04 — Output Parsers

## 🎯 Task
Build a **job description parser** that extracts structured data from raw job posting text.

Output should include:
- `title` (str)
- `company` (str)
- `required_skills` (list of str)
- `salary_range` (str)
- `location` (str)
- `experience_years` (int)
- `remote_ok` (bool)

## Requirements
- Use `PydanticOutputParser` with a typed Pydantic model
- Inject format instructions into the prompt
- Test with 3 different job description texts

## Bonus
- Add a `seniority` field: junior/mid/senior (inferred from description)
- Handle missing fields gracefully (use Optional fields)

## Reference Solution
See `solution.py` after attempting it yourself!
