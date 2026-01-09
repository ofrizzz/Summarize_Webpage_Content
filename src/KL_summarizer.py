import numpy as np

from summarize_building_blocks import SummarizationPipeline

class KLSumSummarizer:
    """
    Implements the greedy KL-Sum strategy using Scikit-Learn for vectorization.
    """
    def __init__(self, target_sentences=3):
        self.target_sentences = target_sentences
        # Lazy import to avoid slow sklearn load at module import time
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(stop_words='english') 

    def summarize(self, sentences: list, max_summary_length: int = 1500) -> list:
        if len(sentences) <= self.target_sentences:
            return sentences

        # 1. Create Bag-of-Words Matrix (Sentences x Vocabulary)
        X = self.vectorizer.fit_transform(sentences)
   
        X = X.toarray() # Convert sparse matrix to dense numpy array
        
        # 2. Calculate Document Distribution (P_doc)
        # Sum all word counts in doc and normalize
        doc_word_counts = X.sum(axis=0)
        p_doc = doc_word_counts / doc_word_counts.sum()

        # 3. Greedy Selection Loop
        selected_indices = []
        summary_word_counts = np.zeros(X.shape[1])
        
        for _ in range(self.target_sentences):
            best_idx = -1
            min_kl_divergence = float('inf')

            # Iterate over all sentences NOT yet selected
            for i in range(len(sentences)):
                if i in selected_indices:
                    continue

                # Temp add sentence to summary
                current_counts = summary_word_counts + X[i]
                
                # Normalize to get P_summary (with smoothing to avoid log(0))
                epsilon = 1e-8
                p_summary = (current_counts + epsilon) / (current_counts.sum() + epsilon * len(current_counts))

                # Calculate KL Divergence: sum(P_doc * log(P_doc / P_summary))
                # We only care about terms where p_doc > 0
                mask = p_doc > 0
                kl_div = np.sum(p_doc[mask] * np.log(p_doc[mask] / p_summary[mask]))

                if kl_div < min_kl_divergence:
                    min_kl_divergence = kl_div
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)
                summary_word_counts += X[best_idx]
            else:
                break  # No more sentences to select                

        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Enforce max summary length
        out = []
        total_summary_length = 0
        for sent in selected_sentences:
            if total_summary_length + len(sent) <= max_summary_length:
                out.append(sent)
                total_summary_length += len(sent)
            else:
                break

        return out

if __name__ == "__main__":
    
    summarizer = KLSumSummarizer(target_sentences=20)
    pipeline = SummarizationPipeline(summarizer)
    
    from load_data import load_summaries_df

    df = load_summaries_df()

    df['kl_summary'] = df['markdown_content'].swifter.apply(
    lambda text: pipeline.run(text, num_sentences=5)
    )
    print(df[['url', 'kl_summary']].head())
    output_data = {
    'data': df.to_dict('records'),
    'url_count': len(df)
    }
    import json
    with open('data/summaries_1k_kl_sum.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
