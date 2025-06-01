import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tokens = ["Ben", "ekmek", "yedim"]

np.random.seed(0)
token_vectors = np.random.rand(len(tokens), 4)
print(token_vectors)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(token_vectors):
    scores = np.dot(token_vectors, token_vectors.T)
    attention_weights = softmax(scores)
    return np.dot(attention_weights, token_vectors), attention_weights

output, weights = self_attention(token_vectors)
print("Attention weights:\n", weights)

sns.heatmap(weights, xticklabels=tokens, yticklabels=tokens, annot=True, cmap="Blues")
plt.title("Self-Attention Weight Matrix")
plt.show()

def causal_attention(token_vectors):
    seq_len = token_vectors.shape[0]
    scores = np.dot(token_vectors, token_vectors.T)

    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    masked_scores = scores + mask

    attention_weights = softmax(masked_scores)
    return np.dot(attention_weights, token_vectors), attention_weights

output_causal, weights_causal = causal_attention(token_vectors)
sns.heatmap(weights_causal, xticklabels=tokens, yticklabels=tokens, annot=True, cmap="Reds")
plt.title("Causal Attention Weight Matrix")
plt.show()

def single_head_attention(Q, K, V):
    scores = np.dot(Q, K.T)
    weights = softmax(scores)
    return np.dot(weights, V), weights

def multi_head_attention(token_vectors, num_heads=2):
    d_model = token_vectors.shape[1]
    assert d_model % num_heads == 0, "Vektör boyutu head sayısına tam bölünmeli"
    
    depth = d_model // num_heads
    heads_output = []
    heads_weights = []

    for i in range(num_heads):
        Q = token_vectors[:, i*depth:(i+1)*depth]
        K = token_vectors[:, i*depth:(i+1)*depth]
        V = token_vectors[:, i*depth:(i+1)*depth]

        out, attn_weights = single_head_attention(Q, K, V)
        heads_output.append(out)
        heads_weights.append(attn_weights)

    concat_output = np.concatenate(heads_output, axis=-1)
    return concat_output, heads_weights

multi_out, multi_weights = multi_head_attention(token_vectors, num_heads=2)

sns.heatmap(multi_weights[0], xticklabels=tokens, yticklabels=tokens, annot=True, cmap="Greens")
plt.title("Multi-Head Attention - Head 1")
plt.show()

sns.heatmap(multi_weights[1], xticklabels=tokens, yticklabels=tokens, annot=True, cmap="Purples")
plt.title("Multi-Head Attention - Head 2")
plt.show()