import numpy as np
import networkx as nx
from summarize_building_blocks import MapReduceSummPipeWeighted, MarkdownCleaner, SmartSplitter, LanguageDetector, normalize_permissive_markdown_headers
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm


class MarkdownCleanerWithPermissiveHeaders(MarkdownCleaner):
    
    def clean(self, text: str) -> str:
        text = normalize_permissive_markdown_headers(text)
        return super().clean(text)


# Splitter - Splits markdown into sentences with weights based on structure
class MarkdownWeightedSplitter:
    def __init__(self):
        self.smart_splitter = SmartSplitter()
    
    def split(self, clean_markdown: str, lang: str) -> tuple[list[str], list[float]]:
        """
        Split markdown into sentences and assign weights based on header hierarchy.
        Returns: (sentences, weights)
        """
        # Split by Header (Structural Splitting)
        headers_to_split_on = [
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3")
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = splitter.split_text(clean_markdown)

        all_sentences = []
        seen_headers = set()

        for section in sections:
            header_values = list(section.metadata.values())
            header_keys = list(section.metadata.keys())

            if header_values:
                current_header = header_keys[-1]
                leaf_header = header_values[-1]
                header_signature = " > ".join(header_values)

                if header_signature not in seen_headers:
                    all_sentences.append({
                        "text": leaf_header,
                        "metadata": section.metadata,
                        "smallest_header": current_header
                    })
                    seen_headers.add(header_signature)

            sents = self.smart_splitter.split(section.page_content, lang)
            for s in sents:
                s_clean = s.strip()
                if len(s_clean) > 10:
                    all_sentences.append({
                        "text": s_clean,
                        "metadata": section.metadata,
                        "smallest_header": None
                    })

        if not all_sentences:
            return [], []

        # Extract texts and assign weights
        texts = [item['text'] for item in all_sentences]
        header_2_weight = {
            "Header 1": 2.0, 
            "Header 2": 1.5, 
            "Header 3": 1.2, 
            None: 1.0
        }
        weights = [header_2_weight[item["smallest_header"]] for item in all_sentences]

        return texts, weights


class WeightedTextRankAlgorithm:
    def __init__(self, model, target_sentences: int = 20):
        self.model = model
        self.target_sentences = target_sentences
    
    def similarity_matrix(self, sentence_embeddings, weights: list[float]):
        """
        Build weighted similarity matrix for TextRank using vectorized operations.
        """
        sim_mat = cosine_similarity(sentence_embeddings)
        positive_sim_mat = (sim_mat + 1) / 2

        # Apply weights using outer product and element-wise multiplication
        # Create weight matrix where element [i,j] = (weights[i] + weights[j])
        weights_array = np.array(weights)
        weight_matrix = np.outer(weights_array, weights_array)
        
        # Element-wise multiplication
        positive_sim_mat *= weight_matrix

        row_sums = positive_sim_mat.sum(axis=1, keepdims=True)
        mat_stochastic = np.divide(
            positive_sim_mat, row_sums, where=row_sums > 1e-6)

        return mat_stochastic

    def summarize(self, sentences: list[str], weights: list[float], max_summary_length: int) -> list[str]:
        """
        Summarize sentences using weighted TextRank algorithm.
        Returns: list of selected sentences
        """
        if len(sentences) <= self.target_sentences:
            return sentences

        sentence_embeddings = self.model.encode(sentences)

        adjacency_matrix = self.similarity_matrix(sentence_embeddings, weights)

        nx_graph = nx.from_numpy_array(adjacency_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Select top N sentences
        selected_sentences = []
        current_length = 0
        
        for i in range(min(self.target_sentences, len(ranked_sentences))):
            sentence = ranked_sentences[i][1]
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed max length
            if current_length + sentence_length > max_summary_length:
                break
                
            selected_sentences.append(sentence)
            current_length += sentence_length

        return selected_sentences
    
if __name__ == "__main__":

    model = SentenceTransformer('all-MiniLM-L6-v2')

    cleaner = MarkdownCleaner()
    detector = LanguageDetector()
    splitter = MarkdownWeightedSplitter()
    algorithm = WeightedTextRankAlgorithm(model=model, target_sentences=20)

    pipeline = MapReduceSummPipeWeighted(
        cleaner=cleaner,
        detector=detector,
        splitter=splitter,
        algorithm=algorithm
    )

    input_path = 'data/summaries_1k.json'
    output_path = 'data/summaries_w_text_rank_map_reduce.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data_list = json_data.get('data', [])
    total_items = len(data_list)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write('{\n  "data": [\n')

        print("Processing and streaming to disk...")
        
        for i, item in enumerate(tqdm(data_list)):
            text = item.get('markdown_content', "")
            
            try:
                generated_summary = pipeline.run(text, 10, 10_000, 1500, False)
            except:
                generated_summary = ""

            
            # Add the new field to the item dictionary
            item['w_textrank_summary'] = generated_summary
            
            json_str = json.dumps(item, ensure_ascii=False, indent=4)
            
            indented_json_str = "\n".join("    " + line for line in json_str.split("\n"))
            f_out.write(indented_json_str)
            
            if i < total_items - 1:
                f_out.write(',\n')
            else:
                f_out.write('\n')
                
            item.clear() 

        # Write the closing of the list and the url_count
        f_out.write('  ],\n')
        f_out.write(f'  "url_count": {total_items}\n')
        f_out.write('}')

    print(f"Done! Saved to {output_path}")