import numpy as np
import networkx as nx
from summarize_building_blocks import MapReduceSummPipe, MarkdownCleaner, SmartSplitter, LanguageDetector
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

class TextRankSummarizer:
    def __init__(self, model, target_sentences: int = 20):
        self.model = model
        self.target_sentences = target_sentences

    def _mat_to_stochastic(self, mat):
        mat_stochastic = (mat + 1) / 2

        row_sums = mat_stochastic.sum(axis=1, keepdims=True)

        mat_stochastic = np.divide(
            mat_stochastic, row_sums, where=row_sums > 1e-6)

        return mat_stochastic

    def summarize(self, sentences, max_summary_length=1500):
        # Split text into sentences

        if len(sentences) <= self.target_sentences:
            return sentences

        # Generate Embeddings (Vectorization)
        sentence_embeddings = self.model.encode(sentences)

        # Create Similarity Matrix
        sim_mat = cosine_similarity(sentence_embeddings)

        nx_graph = nx.from_numpy_array(self._mat_to_stochastic(sim_mat))

        scores = nx.pagerank(nx_graph)

        # Sort sentences by their PageRank score
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Select top N sentences and preserve their original order
        top_sentences = [ranked_sentences[i][1] for i in range(self.target_sentences)]

        # Enforce max summary length
        out = []
        total_summary_length = 0
        for sent in top_sentences:
            if total_summary_length + len(sent) <= max_summary_length:
                out.append(sent)
                total_summary_length += len(sent)
            else:
                break

        return top_sentences
    

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')

    cleaner = MarkdownCleaner()
    detector = LanguageDetector()
    splitter = SmartSplitter()
    algorithm = TextRankSummarizer(model=model, target_sentences=20)

    pipeline = MapReduceSummPipe(
        cleaner=cleaner,
        detector=detector,
        splitter=splitter,
        algorithm=algorithm
    )

    input_path = 'data/summaries_1k.json'
    output_path = 'data/summaries_text_rank_map_reduce.json'

    print("Loading input data...", flush=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data_list = json_data.get('data', [])
    total_items = len(data_list)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write('{\n  "data": [\n')

        print("Processing and streaming to disk...", flush=True)
        
        for i, item in enumerate(tqdm(data_list)):
            text = item.get('markdown_content', "")
            
            try:
                generated_summary = pipeline.run(text, 10, 10_000, 1500, False)
            except:
                generated_summary = ""

            
            item['textrank_summary'] = generated_summary
            
            json_str = json.dumps(item, ensure_ascii=False, indent=4)
            
            indented_json_str = "\n".join("    " + line for line in json_str.split("\n"))
            f_out.write(indented_json_str)
            
            if i < total_items - 1:
                f_out.write(',\n')
            else:
                f_out.write('\n')
                
            item.clear() 

        f_out.write('  ],\n')
        f_out.write(f'  "url_count": {total_items}\n')
        f_out.write('}')

    print(f"Done! Saved to {output_path}")
