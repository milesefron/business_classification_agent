## Introduction
This repo helps business school staff with a common problem: Given an informally-stated company name, match that name to a real, legitimate LinkedIn company profile, and then classify this company among industry types listed in a controled vocabulary. The goal is to automate a tedious, common task: Given that a student has reported going to Company X, we want to add a record of X to our database of known companies. We used LinkedIn profiles as our indicator of what real-world company X corresponds to.

## Do
- Accept company names from the user
- Use web search and world knowledge to match a given company name to a real LinkedIn profile
- Read a list of industries that we know about (i.e. that students wind up working in)
- Match a company name/LinkedIn profile to its most likely industry
- Use OpenAI API functionality to handle all search and inference
- Use JSON and CSV as data exchange formats

## Don't
- Invent new industry class labels. Only use the labels I provide

## Project Structure
- .: python code and configuration goes here
- data: (input and output) goes here
- data/vocab.*.json: different versions of our industry-type controled vocabulary. Only use one per run
- sample_output: examples of the data this software creates