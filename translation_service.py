"""
Station 2: The Diplomat (translation_service.py)
Translates the blueprint's content with high coherence and economy, using context-aware chunking and prompting.
"""

import sys
import copy
import re
from typing import List, Dict, Any, Optional
from document_blueprint import (
    load_blueprint, save_blueprint, find_element_by_id, DocumentBlueprint
)

# --- Placeholder for Gemini API ---
class DummyGeminiAPI:
    def translate(self, prompt: str) -> str:
        # Dummy translation: just returns the prompt for now
        return prompt

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
                    translation_jobs.append({
                        'element_ids': chunk_ids,
                        'source_text': '\n\n'.join(chunk_texts)
                    })
                else:
                    i += 1
    return translation_jobs

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
    
    # Build the structured prompt
    prompt = f"""<System>
You are an expert academic translator, tasked with translating a segment of a document from English to {target_language}. Your response must be of the highest quality, maintaining the original's tone, terminology, and paragraph structure.

**RULES:**
1. Translate ONLY the content within the `<TRANSLATE_THIS>` tag.
2. Do NOT translate the content in the `<CONTEXT_...>` tags. They are for your reference only.
3. The number of paragraphs in your output MUST exactly match the number of paragraphs in the `<TRANSLATE_THIS>` input. A paragraph is separated by a double newline.
4. Do NOT add any formatting, markdown, or commentary. Your output must be ONLY the translated text.
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
</TRANSLATE_THIS>"""
    
    return prompt

def translate_blueprint(
    blueprint_path: str,
    output_path: str = 'translated_blueprint.json',
    target_language: str = 'en',
    gemini_api = None
) -> bool:
    """
    Translates the content of a DocumentBlueprint using context-aware chunking and prompting.
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
        print(f"Translating chunk {i+1}/{len(translation_jobs)}...")
        
        prev_chunk = translation_jobs[i-1] if i > 0 else None
        next_chunk = translation_jobs[i+1] if i < len(translation_jobs)-1 else None
        
        # Build structured prompt
        prompt = build_structured_prompt(
            job, prev_chunk, next_chunk, target_language, blueprint
        )
        
        # Translate
        translated_text = gemini_api.translate(prompt)
        
        # Split back to elements
        translated_parts = split_translated_text(translated_text, len(job['element_ids']))
        
        # Validate paragraph count
        original_paragraphs = [p.strip() for p in job['source_text'].split('\n\n') if p.strip()]
        if len(translated_parts) != len(original_paragraphs):
            print(f"Warning: Paragraph count mismatch in chunk {i+1}. "
                  f"Expected {len(original_paragraphs)}, got {len(translated_parts)}")
        
        # Update elements in translated blueprint
        for eid, translated_paragraph in zip(job['element_ids'], translated_parts):
            el = find_element_in_blueprint(translated_blueprint, eid)
            if el is not None:
                el['content'] = translated_paragraph
    
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