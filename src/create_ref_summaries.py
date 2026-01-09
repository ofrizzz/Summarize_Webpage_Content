import os
import math
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import re
import json

load_dotenv('src/keys.env')  # or load_dotenv() if keys.env is in project root
api_key = os.environ["OPENAI_API_KEY"]
print(f"API Key Loaded: {api_key[:4]}****")  # Print first 4 chars for verification

MODEL = "gpt-5-nano"
MAX_CONTEXT_TOKENS = 400_000

# Safety margin so we don't risk hitting the hard limit (system+prompt+output).
CONTEXT_SAFETY_MARGIN = 30_000

# We need the final output <= 1500 chars, so chunk summaries should be smaller.
MAP_SUMMARY_CHAR_LIMIT = 900
FINAL_SUMMARY_CHAR_LIMIT = 1500


SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes text. Any summaries you provide should be concise "
    "and informative. You will be provided with markdown text scraped from webpages. Focus on "
    "summarizing the main content while omitting the metadata, navigation, and advertisements. "
    "Answer with the summary only, without any additional commentary. If the main content is not "
    "in english, respond with the same language as the input."
)

class MarkdownCleaner:
    """
    Responsible for cleaning Markdown artifacts and removing 'garbage' text
    like navigation menus, code blocks, and footers.
    """
    def __init__(self):
        self.link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')
        self.code_pattern = re.compile(r'```.*?```', re.DOTALL)
        self.header_pattern = re.compile(r'^#+\s+')

    def clean(self, text: str) -> str:
        # Mask code blocks
        text = self.code_pattern.sub("", text)
        
        # Remove image patterns
        text = self.image_pattern.sub("", text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove Markdown headers markers to treat them as text
            # (Optional: You could keep them if you want headers in the summary)
            line = self.header_pattern.sub("", line)

            # Filter Garbage
            if self._is_noise(line):
                continue
            
            # If line passes the filter, replace links with just anchor text
            line = self.link_pattern.sub(r'\1', line)
                
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)

    def _is_noise(self, line: str) -> bool:
        # Filter very short lines (often menu items)
        if len(line.split()) < 4:
            return True
            
        # Filter lines that are mostly links
        links = self.link_pattern.findall(line)
        if links:
            link_len = sum(len(l) for l in links)
            if link_len / len(line) > 0.5:
                return True
                
        return False

def _rough_token_estimate(text: str) -> int:
    """
    Approximate tokens without requiring an external tokenizer.
    A common rule of thumb: 1 token ~= 4 characters in English-ish text.
    For safety, we overestimate slightly.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / 3.6))  # a bit more conservative than /4


def _chunk_by_chars(text: str, max_chars: int, overlap_chars: int = 2_000) -> list[str]:
    """
    Chunk on character length with a small overlap. Tries to cut on paragraph boundaries.
    This is a practical fallback when you don't have a tokenizer available.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        # Try to cut at a paragraph boundary near the end (search backwards).
        # Prefer "\n\n" then "\n" then fallback to hard cut.
        cut = end
        window_start = max(start, end - 8_000)  # only search a tail window for speed
        tail = text[window_start:end]

        idx = tail.rfind("\n\n")
        if idx != -1 and (window_start + idx) > start + 2_000:
            cut = window_start + idx
        else:
            idx = tail.rfind("\n")
            if idx != -1 and (window_start + idx) > start + 2_000:
                cut = window_start + idx

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # advance with overlap
        if cut >= n:
            break
        start = max(0, cut - overlap_chars)

    return chunks


def _call_summary(client: OpenAI, content: str, char_limit: int) -> str:
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT + f" Summary should not exceed {char_limit} characters."},
            {"role": "user", "content": f"Summarize the following content:\n\n##########\n\n{content}"},
        ],
    )
    return (response.output_text or "").strip()


def create_reference_summary(client: OpenAI, content: str) -> str:
    """
    If the content is within context window, summarize directly.
    If it exceeds context window, apply map-reduce:
      - Map: summarize each chunk (smaller char limit)
      - Reduce: summarize concatenated chunk-summaries into final <=1500 chars
    """
    if not content or not content.strip():
        return ""

    # Very rough prompt overhead estimate (system + wrappers). Kept conservative.
    overhead_tokens = _rough_token_estimate(SYSTEM_PROMPT) + 2_000

    content_tokens = _rough_token_estimate(content)
    max_usable_tokens = MAX_CONTEXT_TOKENS - CONTEXT_SAFETY_MARGIN - overhead_tokens

    if content_tokens <= max_usable_tokens:
        return _call_summary(client, content, FINAL_SUMMARY_CHAR_LIMIT)

    # Map-Reduce path
    # Convert max_usable_tokens to a conservative char budget for chunking.
    max_chunk_chars = int(max_usable_tokens * 3.2)  # tokens->chars (conservative)
    chunks = _chunk_by_chars(content, max_chars=max_chunk_chars, overlap_chars=3_000)

    # MAP: summarize each chunk
    chunk_summaries = []
    for i, ch in enumerate(chunks, start=1):
        # Optional: add tiny framing to encourage per-section focus
        ch_with_header = f"[Chunk {i}/{len(chunks)}]\n\n{ch}"
        chunk_summaries.append(_call_summary(client, ch_with_header, MAP_SUMMARY_CHAR_LIMIT))

    combined = "\n\n".join(s for s in chunk_summaries if s).strip()
    if not combined:
        return ""

    # REDUCE: summarize the summaries
    return _call_summary(client, combined, FINAL_SUMMARY_CHAR_LIMIT)



if __name__ == "__main__":
    client = OpenAI(api_key=api_key)
    cleaner = MarkdownCleaner()

    input_path = 'data/summaries_1k.json'
    output_path = 'data/summaries_with_reference.json'

    print("Loading input data...", flush=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    data_list = json_data.get('data', [])
    total_items = len(data_list)
    
    # We open the output file in write mode and manually construct the JSON structure
    # so we can write one item at a time and flush it from memory.
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 1. Write the start of the JSON object
        f_out.write('{\n  "data": [\n')

        print("Processing and streaming to disk...", flush=True)
        
        # Iterate through the list
        for i, item in enumerate(tqdm(data_list)):
            text = item.get('markdown_content', "")
            
            # Generate the summary
            # We call the functions directly on the text string
            cleaned_text = cleaner.clean(text)
            try:
                generated_summary = create_reference_summary(client, cleaned_text)
            except Exception as e:
                generated_summary = ""
            
            # Add the new field to the item dictionary
            item['ref_summary'] = generated_summary
            
            # Write this specific item to the file immediately
            # indent=4 makes it readable, ensure_ascii=False keeps special characters
            json_str = json.dumps(item, ensure_ascii=False, indent=4)
            
            # Indent the whole block to fit inside "data": [ ... ]
            indented_json_str = "\n".join("    " + line for line in json_str.split("\n"))
            f_out.write(indented_json_str)
            
            # If this is not the last item, add a comma
            if i < total_items - 1:
                f_out.write(',\n')
            else:
                f_out.write('\n')
                
            # OPTIONAL: Explicitly clear the 'markdown_content' from the item in memory 
            # if you are extremely constrained (though Python GC usually handles this).
            item.clear() 

        # 2. Write the closing of the list and the url_count
        f_out.write('  ],\n')
        f_out.write(f'  "url_count": {total_items}\n')
        f_out.write('}')

    print(f"Done! Saved to {output_path}")