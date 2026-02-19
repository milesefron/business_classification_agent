# Business Classification Agent

A Python CLI tool that:

1. Resolves a business name to its most likely **LinkedIn company profile**
2. Extracts a short summary of what the organization does
3. Classifies the organization into a **controlled vocabulary**

The system uses OpenAI's **Responses API with Structured Outputs**, along with a two-pass architecture for improved reliability.

---

## Overview

This tool performs classification in two stages:

### Pass 1 — Company Resolution
- Searches for the most likely official LinkedIn company page
- Normalizes the company name
- Produces a short evidence summary (1–3 sentences)

### Pass 2 — Controlled Vocabulary Classification
- Uses only the evidence summary from Pass 1
- Selects exactly one label from your provided vocabulary
- Returns a confidence score and rationale

This separation ensures classification is grounded in retrieved evidence rather than name-based guessing.

---

## Requirements
- Python 3.9+
- OpenAI API key
- Dependencies listed below

---

## Installation

### 1. Create a virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
`pip install openai python-dotenv pydantic tenacity`


## Environment Configuration
To specify your credentials and other details, create a file in this directory called `.env`. You can copy the included dot_env_template file to do this. Just copy the file, calling the copy `.env`. Then edit `.env` so the OPENAI_API_KEY environment variable contains the API key you want to use for this work. 

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1
OPENAI_SEARCH_TOOL=web_search_preview
```

## Vocabulary File
Create a JSON file containing your classification labels:
```
{
  "labels": [
    "Consulting",
    "Finance and Banking",
    "Education",
    "Government and Public Sector",
    "Entrepreneurship",
    "Healthcare and Pharmaceuticals",
    "Marketing and Sales",
    "Operations and Supply Chain",
    "Retail and E-commerce",
    "Technology",
    "Other"
  ]
}
```

## Usage
Run the script, being sure to supply a company name:
'python match_linkedind_and_classify.py "Goldman Sachs" --vocab data/vocab.json'

Optional arguments:
```
--model         Override model (default from OPENAI_MODEL)
--search-tool   Override web search tool
--out           Write output to file instead of stdout
```

Example writing to a file:
```
python match_linkedind_and_classify.py "Amazon" \
  --vocab data/vocab.json \
  --out output.json
```

## Output Format
Example output:
```
{
  "input_name": "Amazon",
  "normalized_name": "Amazon",
  "linkedin_url": "https://www.linkedin.com/company/amazon/",
  "linkedin_display_name": "Amazon",
  "linkedin_confidence": 0.95,
  "evidence_summary": "Amazon is a global technology and e-commerce company offering cloud computing, digital streaming, and online retail services.",
  "classification": {
    "label": "Retail and E-commerce",
    "confidence": 0.87
  },
  "rationale": "Evidence describes Amazon primarily as an online retail platform.",
  "alternates": []
}
```

## Output
The files in `data/sample_output` show the results of a few toy runs. The output files contain quite a few fields, most of which are self-evident. A few are especially relevant, though.
 - `normalized_name` vs `linkedin_display_name`: The first is the version of the name ChatGPT guessed. The second is the version found in the LinkedIn record. So if `linkedin_display_name` is present, this is probably what you want.
 - `linkedin_confidence`: The model's assessment in its confidence that this LinkedIn page is a correct match to your input.
 - `classification`: This shows which of the controled vocab labels the model chose, as well as its confidence for that label.
 - `alternates`: If ChatGPT thinks other LinkedIn pages are plausible matches, they'll be listed here.


## How to Expand / Improve
This program uses a 2-step approach to the problem. In Step 1 we ask OpenAI to match a company name to a LinkedIn profile. Given the output of the first pass, Step 2 tries to assign the company to one of our preexisting class labels. Given this setup, there are two OpenAI/ChatGPT prompts you can tweak to change the model's behavior.

The two prompts are labeled in the code with a comment that says: Prompts.

To find the Step 1 specificially, go to the function called `resolve_prompt`. 

To find the Step 2 prompt, go to the function called `classify_prompt`.

N.B. If you change these prompts, be careful not to change the kinds of data they ask for. For example, the resolve_prompt currently asks for a short summary of about the company. This summary is used downstream, so best not to stop asking for it. But you can safely change most of the wording of these prompts to aim at better results.


## License
Internal use / private project
