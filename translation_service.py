"""
Station 2: The Diplomat (translation_service.py)
Translates the blueprint's content with high coherence and economy, using context-aware chunking and prompting.
"""

import sys
import copy
import re
import json
import os
from typing import List, Dict, Any, Optional
from document_blueprint import (
    load_blueprint, save_blueprint, find_element_by_id, DocumentBlueprint
)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiAPI:
    """A wrapper for the real Google Gemini API."""
    def __init__(self, api_key: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not available. Install with: pip install google-generativeai")
        
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("No API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Use a modern, cost-effective model
        print("✅ Real Gemini API client initialized.")

    def translate(self, prompt: str) -> str:
        """Send prompt to Gemini and return the response."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return ""  # Return empty string on failure to prevent crashes

class DummyGeminiAPI:
    def __init__(self, api_key: str = None):
        print("⚠️  Using DummyGeminiAPI - no actual translation will occur")
        print("   To enable real translation, install google-generativeai and set GEMINI_API_KEY")
    
    def translate(self, prompt: str) -> str:
        # For testing, return a mock JSON response that matches the expected format
        # Extract element IDs from the prompt
        import re
        element_ids = re.findall(r'"([^"]+)"', prompt)
        element_ids = [eid for eid in element_ids if eid.startswith('p') and '_e' in eid]
        
        if not element_ids:
            return '{"translations": {}}'
        
        # Create mock translations
        translations = {}
        for eid in element_ids:
            translations[eid] = f"[MOCK TRANSLATION FOR {eid}]"
        
        return json.dumps({"translations": translations}, indent=2)

def sentence_tokenize(text: str) -> List[str]:
    # Simple sentence tokenizer using regex
    return re.split(r'(?<=[.!?]) +', text.strip())

def chunk_elements(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Implements the semantic chunking algorithm as specified.
    Returns a list of translation jobs: {'element_ids': [...], 'source_text': ...}
    """
    translation_jobs = []
    for page in pages:
        elements = {el['id']: el for el in page['elements']}
        reading_order = page['reading_order']
        i = 0
        while i < len(reading_order):
            eid = reading_order[i]
            el = elements[eid]
            if el['type'] in ['heading_1', 'heading_2', 'heading_3']:
                # Start a new chunk with heading
                chunk_ids = [eid]
                chunk_texts = [el['content']]
                i += 1
                while i < len(reading_order):
                    next_eid = reading_order[i]
                    next_el = elements[next_eid]
                    if next_el['type'] in ['heading_1', 'heading_2', 'heading_3', 'table', 'image']:
                        break
                    if next_el['type'] in ['paragraph', 'list_item', 'caption']:
                        chunk_ids.append(next_eid)
                        chunk_texts.append(next_el['content'])
                    i += 1
                print(f"[DEBUG CHUNK]: Created heading-based chunk with IDs {chunk_ids}. Content length: {len(''.join(chunk_texts))}")
                translation_jobs.append({
                    'element_ids': chunk_ids,
                    'source_text': '\n\n'.join(chunk_texts)
                })
            else:
                # No heading: group consecutive paragraphs
                chunk_ids = []
                chunk_texts = []
                count = 0
                while i < len(reading_order) and count < 4:
                    eid = reading_order[i]
                    el = elements[eid]
                    if el['type'] in ['paragraph', 'list_item']:
                        chunk_ids.append(eid)
                        chunk_texts.append(el['content'])
                        count += 1
                        i += 1
                    else:
                        break
                if chunk_ids:
                    print(f"[DEBUG CHUNK]: Created paragraph-based chunk with IDs {chunk_ids}. Content length: {len(''.join(chunk_texts))}")
                    translation_jobs.append({
                        'element_ids': chunk_ids,
                        'source_text': '\n\n'.join(chunk_texts)
                    })
                else:
                    i += 1
    return translation_jobs

def parse_json_translation_response(response: str) -> Dict[str, str]:
    """
    Parses the JSON response from the LLM and extracts translations.
    Returns a dictionary mapping element_ids to translated content.
    """
    try:
        # Clean the response - remove any markdown formatting
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Parse JSON
        data = json.loads(cleaned_response)
        
        if "translations" in data and isinstance(data["translations"], dict):
            return data["translations"]
        else:
            print(f"Warning: Invalid JSON structure in response: {response}")
            return {}
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return {}
    except Exception as e:
        print(f"Unexpected error parsing translation response: {e}")
        return {}

def split_translated_text(translated_text: str, num_elements: int) -> List[str]:
    # Split by double newlines, pad if needed
    parts = [p.strip() for p in translated_text.split('\n\n') if p.strip()]
    if len(parts) < num_elements:
        parts += [''] * (num_elements - len(parts))
    return parts[:num_elements]

def find_element_in_blueprint(blueprint: DocumentBlueprint, element_id: str) -> Optional[Dict[str, Any]]:
    # Wrapper for find_element_by_id
    return find_element_by_id(blueprint, element_id)

def create_structural_hints(job: Dict[str, Any], blueprint: DocumentBlueprint) -> str:
    """Creates structural hints for the translation prompt."""
    hints = []
    for element_id in job['element_ids']:
        element = find_element_in_blueprint(blueprint, element_id)
        if element:
            hints.append(f"'{element['type']}'")
    
    if len(hints) == 1:
        return f"This chunk contains one {hints[0]} element."
    else:
        return f"This chunk contains {len(hints)} elements: {', '.join(hints)}."

def build_structured_prompt(
    current_chunk: Dict[str, Any],
    previous_chunk: Optional[Dict[str, Any]],
    next_chunk: Optional[Dict[str, Any]],
    target_language: str,
    blueprint: DocumentBlueprint
) -> str:
    """
    Builds the rigorous structured prompt as specified in the requirements.
    """
    # Extract context
    prompt_context_previous = ""
    if previous_chunk:
        prev_sentences = sentence_tokenize(previous_chunk['source_text'])
        if prev_sentences:
            prompt_context_previous = prev_sentences[-1]
    
    prompt_context_next = ""
    if next_chunk:
        next_sentences = sentence_tokenize(next_chunk['source_text'])
        if next_sentences:
            prompt_context_next = next_sentences[0]
    
    # Create structural hints
    structural_hints = create_structural_hints(current_chunk, blueprint)
    
    # Build the structured prompt with JSON contract
    prompt = f"""<System>
You are an expert academic translator, tasked with translating a segment of a document from English to {target_language}. Your response must be of the highest quality, maintaining the original's tone, terminology, and paragraph structure.

**CRITICAL RULES:**
1. You MUST respond with ONLY a valid JSON object.
2. The JSON must have the exact structure shown in the <OUTPUT_FORMAT> section.
3. Each element_id must be mapped to its translated content.
4. Do NOT add any formatting, markdown, or commentary outside the JSON.
5. Do NOT translate the content in the `<CONTEXT_...>` tags. They are for your reference only.
</System>

<ContextualInformation>
    <PrecedingText>
        {prompt_context_previous}
    </PrecedingText>
    <FollowingText>
        {prompt_context_next}
    </FollowingText>
    <StructuralHints>
        {structural_hints}
    </StructuralHints>
</ContextualInformation>

<TRANSLATE_THIS>
{current_chunk['source_text']}
</TRANSLATE_THIS>

<OUTPUT_FORMAT>
{{
    "translations": {{
        {', '.join([f'"{element_id}": "translated_content_here"' for element_id in current_chunk['element_ids']])}
    }}
}}
</OUTPUT_FORMAT>"""
    
    return prompt

def translate_blueprint(
    blueprint_path: str,
    output_path: str = 'translated_blueprint.json',
    target_language: str = 'en',
    gemini_api = None,
    strict_json_contract: bool = True
) -> bool:
    """
    Translates the content of a DocumentBlueprint using context-aware chunking and prompting.
    If strict_json_contract is True, fail if the LLM does not return valid JSON.
    """
    blueprint = load_blueprint(blueprint_path)
    if not blueprint:
        print(f"Failed to load blueprint: {blueprint_path}")
        return False
    
    translated_blueprint = copy.deepcopy(blueprint)
    translation_jobs = chunk_elements(blueprint['pages'])
    
    if gemini_api is None:
        gemini_api = DummyGeminiAPI()
    
    print(f"Processing {len(translation_jobs)} translation chunks...")
    
    for i, job in enumerate(translation_jobs):
        print(f"Translating chunk {i+1}/{len(translation_jobs)} with {len(job['element_ids'])} elements...")
        source_text_to_translate = job.get('source_text', '').strip()
        if not source_text_to_translate:
            print(f"⚠️  WARNING: Skipping chunk {i+1} because it contains no text content.")
            continue
        prev_chunk = translation_jobs[i-1] if i > 0 else None
        next_chunk = translation_jobs[i+1] if i < len(translation_jobs)-1 else None
        
        # Build structured prompt
        prompt = build_structured_prompt(
            job, prev_chunk, next_chunk, target_language, blueprint
        )
        
        # Translate
        translated_text = gemini_api.translate(prompt)
        
        # Parse JSON response instead of splitting text
        translations = parse_json_translation_response(translated_text)
        
        if not translations:
            print(f"❌ Failed to parse JSON response for chunk {i+1}.")
            if strict_json_contract:
                print(f"❌ Strict mode: Aborting translation due to invalid JSON contract from LLM.")
                return False
            else:
                print(f"⚠️  Fallback: Using text splitting, but this is NOT recommended.")
                translated_parts = split_translated_text(translated_text, len(job['element_ids']))
                for eid, translated_paragraph in zip(job['element_ids'], translated_parts):
                    el = find_element_in_blueprint(translated_blueprint, eid)
                    if el is not None:
                        el['content'] = translated_paragraph
        else:
            # Use JSON-based translations
            print(f"Successfully parsed JSON response for chunk {i+1}")
            for element_id in job['element_ids']:
                if element_id in translations:
                    el = find_element_in_blueprint(translated_blueprint, element_id)
                    if el is not None:
                        el['content'] = translations[element_id]
                else:
                    print(f"Warning: Missing translation for element {element_id} in chunk {i+1}")
        
        # Validate translation coverage
        if len(translations) != len(job['element_ids']):
            print(f"Warning: Translation coverage mismatch in chunk {i+1}. "
                  f"Expected {len(job['element_ids'])}, got {len(translations)}")
    
    # Update language in metadata
    translated_blueprint['metadata']['language'] = target_language
    
    # Save translated blueprint
    save_blueprint(translated_blueprint, output_path)
    print(f"Translated blueprint saved to {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translation_service.py <document_blueprint.json> [output.json] [target_language]")
        sys.exit(1)
    blueprint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'translated_blueprint.json'
    target_language = sys.argv[3] if len(sys.argv) > 3 else 'en'
    translate_blueprint(blueprint_path, output_path, target_language) 