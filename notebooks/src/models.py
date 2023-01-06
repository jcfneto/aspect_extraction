import tensorflow as tf

from dataclasses import dataclass

from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen, output_dim=embed_dim
        )

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


@dataclass
class AEModelConfig:
    num_tags: int
    vocab_size: int
    maxlen: int = 128
    embed_dim: int = 32
    num_heads: int = 2
    ff_dim: int = 32


class AEModel(keras.Model):

    def __init__(self, config: AEModelConfig):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(
            config.maxlen, config.vocab_size, config.embed_dim
        )
        self.transformer_block = TransformerBlock(
            config.embed_dim, config.num_heads, config.ff_dim
        )
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(config.ff_dim, activation='relu')
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(config.num_tags, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=False)
        x = self.ff(x)
        x = self.dropout2(x, training=False)
        x = self.ff_final(x)
        return x


class CustomNonPaddingTokenLoss(keras.losses.Loss):

    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
