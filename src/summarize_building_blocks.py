import re
import pysbd
from langdetect import detect, LangDetectException
# ==========================================
# Pre-processing Module (Swappable)
# ==========================================
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

def normalize_permissive_markdown_headers(markdown_text: str) -> str:
    """
    Converts Setext-style headers to ATX (#), even if there are blank lines 
    between the text and the underline.
    """
    # 1. Matches Level 1 (Underlined with =)
    # \n\s* -> Matches the newline after text, PLUS any extra blank lines before the ===
    pattern_h1 = re.compile(r'^[ \t]*(.+?)[ \t]*\n\s*={3,}[ \t]*$', re.MULTILINE)

    # 2. Matches Level 2 (Underlined with -)
    pattern_h2 = re.compile(r'^[ \t]*(.+?)[ \t]*\n\s*-{3,}[ \t]*$', re.MULTILINE)

    # Replacement lambda to strip whitespace from the captured title
    text_h1 = pattern_h1.sub(lambda m: f"# {m.group(1).strip()}", markdown_text)
    text_h2 = pattern_h2.sub(lambda m: f"## {m.group(1).strip()}", text_h1)

    return text_h2

# ==========================================
# Language Detection Module (Swappable)
# ==========================================
class LanguageDetector:
    """
    Wrapper for language detection. 
    Can be swapped for 'fasttext' for higher performance.
    """
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en" # Fallback

# ==========================================
# Sentence Splitter Module
# ==========================================
class SmartSplitter:
    """
    Handles the complexity of splitting text into sentences 
    while respecting language-specific rules (abbreviations, etc).
    """
    def __init__(self):
        self._cache = {}

    def split(self, text: str, lang_code: str):
        # PySBD initialization can be heavy, so we cache segmenters per language
        if lang_code not in self._cache:
            # Fallback for languages PySBD doesn't support explicitly
            try:
                self._cache[lang_code] = pysbd.Segmenter(language=lang_code, clean=True)
            except:
                self._cache[lang_code] = pysbd.Segmenter(language='en', clean=True)
        
        return self._cache[lang_code].segment(text)

# ==========================================
# Pipeline Orchestrator
# ==========================================
class SummarizationPipeline:
    def __init__(self, summarizer):
        self.cleaner = MarkdownCleaner()
        self.detector = LanguageDetector()
        self.splitter = SmartSplitter()
        self.algorithm = summarizer

    def run(self, raw_markdown: str, num_sentences: int = 3) -> str:
        # Update config
        self.algorithm.target_sentences = num_sentences

        clean_text = self.cleaner.clean(raw_markdown)

        lang = self.detector.detect_language(clean_text)

        sentences = self.splitter.split(clean_text, lang)

        summary_sentences = self.algorithm.summarize(sentences)

        return "\n".join(summary_sentences)

# ==========================================
# Map-Reduce Summarization Pipeline
# ==========================================
class MapReduceSummPipe:
    def __init__(self, cleaner, detector, splitter, algorithm):
        self.cleaner = cleaner
        self.detector = detector
        self.splitter = splitter
        self.algorithm = algorithm

    def chunk_content_with_overlap(self, sentences: list[str], max_chunk_length: int) -> list[list[str]]:
        """
        Split sentences into chunks where each chunk's total length < max_chunk_length.
        Adjacent chunks overlap by 1 sentence.
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length >= max_chunk_length:
                # Save current chunk
                chunks.append(current_chunk)
                # Start new chunk with the last sentence from previous chunk (overlap)
                last_sentence = current_chunk[-1]
                current_chunk = [last_sentence, sentence]
                current_length = len(last_sentence) + sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def chunk_content_no_overlap(self, sentences: list[str], max_chunk_length: int) -> list[list[str]]:
        """
        Split sentences into chunks where each chunk's total length < max_chunk_length.
        No overlap between adjacent chunks.
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length >= max_chunk_length and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def run_map_reduce_summarization(self, sentences: list[str], max_chunk_length: int, max_summary_length: int, chunk_with_overlap: bool) -> str:
        """
        If the content is within threshold, summarize directly.
        If it exceeds context window, apply map-reduce:
        - Map: summarize each chunk (smaller char limit)
        - Reduce: summarize concatenated chunk-summaries into final <=1500 chars
        """
        
        # Map-Reduce path
        if chunk_with_overlap:
            chunks = self.chunk_content_with_overlap(sentences, max_chunk_length)
        else:
            chunks = self.chunk_content_no_overlap(sentences, max_chunk_length)

        if len(chunks) == 1:
            return self.algorithm.summarize(chunks[0], max_summary_length)

        chunk_summaries_sentences = []
        for chunk in chunks:
            chunk_summaries_sentences.append(self.algorithm.summarize(chunk, max_summary_length))
        
        # flatten list of list of sentences to one sentences list:
        combined_summary_sentences = [sentence for summary in chunk_summaries_sentences for sentence in summary]

        return self.algorithm.summarize(combined_summary_sentences, max_summary_length)
    

    def run(self, raw_markdown: str, num_sentences: int = 20, max_chunk_length: int = 1000, max_summary_length: int = 1500, chunk_with_overlap: bool = False) -> str:
        # Update config
        self.algorithm.target_sentences = num_sentences

        # Step 1: Clean
        clean_text = self.cleaner.clean(raw_markdown)

        # Step 2: Detect Language
        lang = self.detector.detect_language(clean_text)

        # Step 3: Split
        sentences = self.splitter.split(clean_text, lang)


        # Step 4: Summarize
        summary_sentences = self.run_map_reduce_summarization(sentences, max_chunk_length=max_chunk_length, max_summary_length=max_summary_length, chunk_with_overlap=chunk_with_overlap)

        return "\n".join(summary_sentences)
    

# ==========================================
# Map-Reduce Summarization Pipeline with Weights
# ==========================================
class MapReduceSummPipeWeighted:
    def __init__(self, cleaner, detector, splitter, algorithm):
        self.cleaner = cleaner
        self.detector = detector
        self.splitter = splitter
        self.algorithm = algorithm

    def chunk_content_with_overlap(self, sentences: list[str], weights: list[float], max_chunk_length: int) -> tuple[list[list[str]], list[list[float]]]:
        """
        Split sentences into chunks where each chunk's total length < max_chunk_length.
        Adjacent chunks overlap by 1 sentence.
        Returns both sentence chunks and corresponding weight chunks.
        """
        if not sentences:
            return [], []
        
        sentence_chunks = []
        weight_chunks = []
        current_sentence_chunk = [sentences[0]]
        current_weight_chunk = [weights[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            weight = weights[i]
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length >= max_chunk_length:
                # Save current chunk
                sentence_chunks.append(current_sentence_chunk)
                weight_chunks.append(current_weight_chunk)
                # Start new chunk with the last sentence from previous chunk (overlap)
                last_sentence = current_sentence_chunk[-1]
                last_weight = current_weight_chunk[-1]
                current_sentence_chunk = [last_sentence, sentence]
                current_weight_chunk = [last_weight, weight]
                current_length = len(last_sentence) + sentence_length
            else:
                # Add sentence to current chunk
                current_sentence_chunk.append(sentence)
                current_weight_chunk.append(weight)
                current_length += sentence_length
        
        # Add the last chunk
        if current_sentence_chunk:
            sentence_chunks.append(current_sentence_chunk)
            weight_chunks.append(current_weight_chunk)
        
        return sentence_chunks, weight_chunks

    def chunk_content_no_overlap(self, sentences: list[str], weights: list[float], max_chunk_length: int) -> tuple[list[list[str]], list[list[float]]]:
        """
        Split sentences into chunks where each chunk's total length < max_chunk_length.
        No overlap between adjacent chunks.
        Returns both sentence chunks and corresponding weight chunks.
        """
        if not sentences:
            return [], []
        
        sentence_chunks = []
        weight_chunks = []
        current_sentence_chunk = []
        current_weight_chunk = []
        current_length = 0
        
        for sentence, weight in zip(sentences, weights):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the limit
            if current_length + sentence_length >= max_chunk_length and current_sentence_chunk:
                # Save current chunk and start new one
                sentence_chunks.append(current_sentence_chunk)
                weight_chunks.append(current_weight_chunk)
                current_sentence_chunk = [sentence]
                current_weight_chunk = [weight]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_sentence_chunk.append(sentence)
                current_weight_chunk.append(weight)
                current_length += sentence_length
        
        # Add the last chunk
        if current_sentence_chunk:
            sentence_chunks.append(current_sentence_chunk)
            weight_chunks.append(current_weight_chunk)
        
        return sentence_chunks, weight_chunks

    def run_map_reduce_summarization(self, sentences: list[str], weights: list[float], max_chunk_length: int, max_summary_length: int, chunk_with_overlap: bool) -> str:
        """
        If the content is within threshold, summarize directly.
        If it exceeds context window, apply map-reduce:
        - Map: summarize each chunk (smaller char limit)
        - Reduce: summarize concatenated chunk-summaries into final <=1500 chars
        """
        
        # Map-Reduce path
        if chunk_with_overlap:
            sentence_chunks, weight_chunks = self.chunk_content_with_overlap(sentences, weights, max_chunk_length)
        else:
            sentence_chunks, weight_chunks = self.chunk_content_no_overlap(sentences, weights, max_chunk_length)

        if len(sentence_chunks) == 1:
            return self.algorithm.summarize(sentence_chunks[0], weight_chunks[0], max_summary_length)

        chunk_summaries_sentences = []
        chunk_summaries_weights = []
        for sentence_chunk, weight_chunk in zip(sentence_chunks, weight_chunks):
            summary = self.algorithm.summarize(sentence_chunk, weight_chunk, max_summary_length)
            chunk_summaries_sentences.append(summary)
            # For intermediate summaries, assign uniform weights
            chunk_summaries_weights.append([1.0] * len(summary))
        
        # flatten list of list of sentences to one sentences list:
        combined_summary_sentences = [sentence for summary in chunk_summaries_sentences for sentence in summary]
        combined_summary_weights = [weight for weights in chunk_summaries_weights for weight in weights]

        return self.algorithm.summarize(combined_summary_sentences, combined_summary_weights, max_summary_length)
    

    def run(self, raw_markdown: str, num_sentences: int = 20, max_chunk_length: int = 10_000, max_summary_length: int = 1500, chunk_with_overlap: bool = False) -> str:
        # Update config
        self.algorithm.target_sentences = num_sentences

        # Step 1: Clean
        clean_text = self.cleaner.clean(raw_markdown)

        # Step 2: Detect Language
        lang = self.detector.detect_language(clean_text)

        # Step 3: Split - now returns both sentences and weights
        sentences, weights = self.splitter.split(clean_text, lang)


        # Step 4: Summarize
        summary_sentences = self.run_map_reduce_summarization(sentences, weights, max_chunk_length=max_chunk_length, max_summary_length=max_summary_length, chunk_with_overlap=chunk_with_overlap)

        return "\n".join(summary_sentences)