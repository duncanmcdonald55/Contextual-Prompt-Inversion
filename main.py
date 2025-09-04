import faiss
import openpyxl
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def turn_to_int(item):
    if item == "Neutral":
        return item
    return int(item)

def wordcloud_gen(freq, filename):

    color_map = {
        'core_content': 'black',
        'style': 'darkcyan',
        'execution': 'indigo'
    }

    all_words = []
    word_to_category_map = {}

    for category, results in freq.items():

        for text, score in results:

            all_words.append((text, score))
            word_to_category_map[text] = category

    frequencies = {}
    for text, score in all_words:
        if text not in frequencies or score > frequencies[text]:
            frequencies[text] = score
    if not frequencies:
        return
    
    def category_color_func(word, font_size, position, orientation, random_state = None, **kwargs):
        category = word_to_category_map.get(word)
        return color_map.get(category, 'grey')
    
    wc = WordCloud(
        width = 1200, height = 600,
        background_color='white',
        collocations = False,
        color_func=category_color_func
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(15,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to {filename}")

class PromptDecomposerMLP(nn.Module):
    def __init__(self, input_size, shared_hidden_size, output_embedding_size, dropout_rate = 0.3):
        super(PromptDecomposerMLP, self).__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_hidden_size, shared_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.content_head = nn.Linear(shared_hidden_size // 2, output_embedding_size)
        self.execution_head = nn.Linear(shared_hidden_size // 2, output_embedding_size)
        self.style_head = nn.Linear(shared_hidden_size // 2, output_embedding_size)

    def forward(self, x):
        shared_features = self.shared_trunk(x)
        content_out = self.content_head(shared_features)
        exec_out = self.execution_head(shared_features)
        style_out = self.style_head(shared_features)
        return content_out, exec_out, style_out
    


INDEX_PATH_CONTENT = "faiss_indexes/content.index"
INDEX_PATH_STYLE = "faiss_indexes/style.index"
INDEX_PATH_EXEC = "faiss_indexes/execution.index"

FILTER_STYLE = "filtering/categorized_stylistic_terms.csv"
FILTER_EXECUTION = "filtering/categorized_execution_terms.csv"

DATA_PATH_CONTENT_EXCEL = 'faiss_terms/content.xlsx'
DATA_PATH_STYLE_EXCEL = 'faiss_terms/stylistic.xlsx'
DATA_PATH_EXECUTION_EXCEL = 'faiss_terms/execution.xlsx'

SBERT_MODEL = "all-mpnet-base-v2"
MLP_MODEL_PATH = "best_model.pth"
OUTPUT_EMBEDDING_SIZE = 768
SHARED_HIDDEN_SIZE = 1024

print("Loading SBERT Model")
model = SentenceTransformer(SBERT_MODEL)
embedding_dim = model.get_sentence_embedding_dimension()

print("- Loading PromptDecomposerMLP...")
mlp_model = PromptDecomposerMLP(input_size=embedding_dim, shared_hidden_size=SHARED_HIDDEN_SIZE, output_embedding_size=OUTPUT_EMBEDDING_SIZE)
mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH))
mlp_model.eval()

print("Loading FAISS Indexes")
index_content = faiss.read_index(INDEX_PATH_CONTENT)
index_exec = faiss.read_index(INDEX_PATH_EXEC)
index_style = faiss.read_index(INDEX_PATH_STYLE)

print("Loading text data from XLSX")
df_content = pd.read_excel(DATA_PATH_CONTENT_EXCEL)
content_texts = df_content["item"].to_list()

df_execution = pd.read_excel(DATA_PATH_EXECUTION_EXCEL)
exec_texts = df_execution["item"].to_list()

df_style = pd.read_excel(DATA_PATH_STYLE_EXCEL)
style_texts = df_style["item"].to_list()

print("Loading filtering style and execution tables")
filter_style = pd.read_csv(FILTER_STYLE)
style_f = filter_style["category"].to_list()

filter_execution = pd.read_csv(FILTER_EXECUTION)
exec_f = filter_execution["category"].to_list()
exec_year = filter_execution["year"].apply(turn_to_int).to_list()

print("Assets loaded successfully!")


def find_matches(query_text, k_results = 20, model_gen = None, year_gen = None):

    k_results_content = 10
    sbert_embedding = model.encode([query_text])[0]
    sbert_tensor = torch.from_numpy(sbert_embedding).float()
    with torch.no_grad():
        content_pred, execution_pred, style_pred = mlp_model(sbert_tensor)
    
    vec_content = np.array([content_pred.numpy()]).astype('float32')
    vec_execution = np.array([execution_pred.numpy()]).astype('float32')
    vec_style = np.array([style_pred.numpy()]).astype('float32')

    index_content.nprobe = 8
    index_exec.nprobe = 8
    index_style.nprobe = 8

    _, indices_content = index_content.search(vec_content, k_results_content)
    core_matches = []
    for idx in indices_content[0]:
        if idx != -1:
            retrieved_vector = index_content.reconstruct(int(idx))
            similarity = util.cos_sim(vec_content, retrieved_vector.reshape(1,-1))
            core_matches.append((content_texts[idx], similarity.item()))

    _, indices_execution = index_exec.search(vec_execution, k_results)
    execution_matches = []
    for idx in indices_execution[0]:
        if idx != -1:
            retrieved_vector = index_exec.reconstruct(int(idx))
            similarity = util.cos_sim(vec_execution, retrieved_vector.reshape(1,-1))
            if model_gen:
                if model_gen.lower() == "stable diffusion" and exec_f[idx] == "Technical":
                    similarity *= 1.1
                elif model_gen.lower() == "dalle" and exec_f[idx] == "Photographic":
                    similarity *= 1.1
                elif model_gen.lower() == "midjourney" and exec_f[idx] == "Artistic":
                    similarity *= 1.1
                else:
                    pass
            if year_gen:
                if year_gen == 2022 and exec_year[idx] == 2022:
                    similarity *= 1.05
                elif year_gen == 2023 and exec_year[idx] == 2023:
                    similarity *= 1.05
                else:
                    pass
            execution_matches.append((exec_texts[idx], similarity.item()))

    _, indices_style = index_style.search(vec_style, k_results)
    style_matches = []
    for idx in indices_style[0]:
        if idx != -1:
            retrieved_vector = index_style.reconstruct(int(idx))
            similarity = util.cos_sim(vec_style, retrieved_vector.reshape(1,-1))
            if model_gen:
                if model_gen.lower() == "stable diffusion" and style_f[idx] == "Technical":
                    similarity *= 1.5
                elif model_gen.lower() == "dalle" and style_f[idx] == "Photographic":
                    similarity *= 1.5
                elif model_gen.lower() == "midjourney" and style_f[idx] == "Artistic":
                    similarity *= 1.5
                else:
                    pass
            style_matches.append((style_texts[idx], similarity.item()))

    sorted_exec_matches = sorted(execution_matches, key=lambda x: x[1], reverse=True)
    sorted_style_matches = sorted(style_matches, key=lambda x: x[1], reverse=True)


    results = {
        'core_content': core_matches,
        'style': sorted_style_matches[:10],
        'execution': sorted_exec_matches[:10]
    }
    return results

if __name__ == "__main__":
    my_query = """[CAPTION] "A somber procession. The weight of justice, or an unprecedented political moment?" [HASHTAGS] #politicalnews #breakingnews #justice #legalprocess #currentevents [COMMENTS] This image speaks volumes.|An iconic moment in history.|The expressions tell a story.|Powerful and thought-provoking.|What are your thoughts on this? """
    
    matches = find_matches(my_query, k_results = 20)

    print("\n--- Top 10 Results ---")
    print("\nCore Content Matches:")
    for i, (text, score) in enumerate(matches['core_content']):
        print(f" {i+1}. {text} (Similarity: {score:.4f})")

    print("\nStylistic Modifier Matches:")
    for i, (text, score) in enumerate(matches["style"]):
        print(f" {i+1}. {text} (Similarity: {score:.4f})")

    print("\nExecution Modifier Matches:")
    for i, (text, score) in enumerate(matches["execution"]):
        print(f" {i+1}. {text} (Similarity: {score:.4f})")
    
    print("Collated Word Cloud")
    wordcloud_gen(matches, "collated_word_cloud.png")


