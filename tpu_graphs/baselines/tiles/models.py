# Copyright 2023 The tpu_graphs Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines GNNs and MLP models for ranking module configurations on tiles data.

The high-level models are:
  + LateJoinResGCN: Applies GNN on op nodes. The GNN output will be concatenated
    with module config features. Finally, MLP outputs scalar that ranks each
    config. Here, GNN is GCN with residual connections.
  + EarlyJoinResGCN: Like above, however, it replicates (==broadcasts) module
    config features on op nodes then applies ResGCN, then applies MLP.
  + EarlyJoinSAGE and LateJoinSAGE: like above, but using GraphSAGE as backbone.

[GCN] Kipf and Welling, ICLR'17.
[GraphSAGE] Hamilton et al, NeurIPS'17.
"""
import abc

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import layers

from tpu_graphs.baselines.tiles import implicit


class _ConfigFeatureJoiner(abc.ABC):
  """Defines interface for joining config features with op nodes.

  The implementations join features pre- or post-GNN, respectively, named as
  `_EarlyJoin` and `_LateJoin`.
  """

  @abc.abstractmethod
  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    """Should return feature matrix (or tensor) of op-nodes."""
    raise NotImplementedError()

  def get_penultimate_output(
      self, pooled: tf.Tensor, unused_graph: tfgnn.GraphTensor,
      unused_num_configs: int) -> tf.Tensor:
    """Must return tensor with shape `[batch_size, num_configs, hidden_dim]`."""
    return pooled


def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)


class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})


class _SAGE(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GraphSAGE GNN Backbone."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, final_mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 64):
    super().__init__()
    self._num_configs = num_configs
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gnn_layers = []
    for unused_i in range(num_gnns):
      self._gnn_layers.append(_mlp([hidden_dim], hidden_activation))
    self._postnet = _mlp(
        [hidden_dim] * final_mlp_layers + [1], hidden_activation)
    self._activation_fn = getattr(tf.nn, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)
    bidirectional_adj = implicit.AdjacencyMultiplier(graph, 'feed')
    bidirectional_adj = implicit.Sum(
        bidirectional_adj, bidirectional_adj.transpose())
    for gnn_layer in self._gnn_layers:
      y = bidirectional_adj @ x
      y = tf.concat([y, x], axis=-1)
      y = gnn_layer(y)
      y = self._activation_fn(y)
      y = tf.nn.l2_normalize(y, axis=-1)
      x = y

    pooled = tfgnn.pool_nodes_to_context(graph, 'op', 'sum', feature_value=x)

    pooled = self.get_penultimate_output(pooled, graph, num_configs)
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    # _postnet maps across last channel from hidden_dim to 1.

    return tf.squeeze(self._postnet(pooled), -1)


class _ResGCN(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GCN backbone with residual connections."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 32, directed: bool = False,
               reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gc_layers = []
    self._activation_fn = getattr(tf.nn, hidden_activation)
    self._directed = directed
    self._reduction = reduction
    self._prenet = _mlp([hidden_dim, hidden_dim], self._activation_fn)
    self._postnet = _mlp([hidden_dim, 1], self._activation_fn)
    for _ in range(num_gnns):
      if directed:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn))
      else:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),)
      self._gc_layers.append(tuple(configs_mlps))

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)

    am = implicit.AdjacencyMultiplier(graph, 'feed')
    am = am.add_eye().normalize_right()
    x = self._prenet(x)
    for gc_layer in self._gc_layers:
      y = self._activation_fn(x)
      forward_layer = gc_layer[0]
      if self._directed:
        reverse_layer = gc_layer[1]
        self_layer = gc_layer[2]
        y = (forward_layer(am @ y) + reverse_layer(am.transpose() @ y)
             + self_layer(y))
      else:
        y = forward_layer((am @ y) + (am.transpose() @ y)  + y)

      # Residual connection.
      x += y

    x = self._activation_fn(x)
    pooled = tfgnn.pool_nodes_to_context(
        graph, 'op', self._reduction, feature_value=x)
    # Pooled has shape [batch_size, num_configs, hidden_dim]

    pooled = self.get_penultimate_output(pooled, graph, num_configs)

    return tf.squeeze(self._postnet(pooled), -1)


class _EarlyJoin(_ConfigFeatureJoiner):
  """Joins module configuration features before applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    graph = _EarlyJoin.attach_config_features_on_op_nodes_imsi(graph)
    return tf.concat([
        # Shape (num_nodes, num_configs, embedding dim)
        tf.stack([graph.node_sets['op']['op_e']] * num_configs, 1),
        # Shape (num_nodes, num_configs, config feat dim)
        graph.node_sets['op']['config_feats'],
        # Shape (num_nodes, num_configs, op feat dim)
        tf.stack([graph.node_sets['op']['feats']] * num_configs, 1),
    ], axis=-1)

  @staticmethod
  def attach_config_features_on_op_nodes_imsi(
      graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """Replicates config features on every op node."""
    # Shape: [batch_size * num_configs, feature size].
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # shape: (total number of op nodes, config feats dimension)
    op_broadcasted = tfgnn.broadcast_node_to_edges(
        graph, 'g_op', tfgnn.SOURCE, feature_value=config_feats)
    op_features = dict(graph.node_sets['op'].features)
    op_features['config_feats'] = op_broadcasted
    return graph.replace_features(node_sets={'op': op_features})

from tensorflow.keras.layers import MultiHeadAttention

class _EarlyJoinWithAttention(_EarlyJoin):
  def attach_config_features_on_op_nodes(self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    graph = self.attach_config_features_on_op_nodes_imsi(graph)
    op_features = self.get_op_node_features(graph, num_configs)
    attention_output = MultiHeadAttention(num_heads=3, key_dim=5, query=op_features, value=op_features, key=op_features)
    combined_features = tf.concat([op_features, attention_output], axis=-1)
    return combined_features
  

#####################################################
class _Early_LateJoin(_ConfigFeatureJoiner):
  """Joins module configuration features before applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    config_feats = graph.node_sets['config']['feats']  #= (100,21) 
    batch_size = graph.node_sets['config'].sizes.shape[0] #= (10) 
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    #= (10,10,21) 
    
    
    # shape: (total number of op nodes, config feats dimension)
    op_broadcasted = tfgnn.broadcast_node_to_edges(
        graph, 'g_op', tfgnn.SOURCE, feature_value=config_feats)
    # (369,10,21)
    op_features = dict(graph.node_sets['op'].features) 
    op_features['config_feats'] = op_broadcasted   # add the new key config_feats and its value
    graph = graph.replace_features(node_sets={'op': op_features})

    # just concatenate each feature(embedding, config, node feature)
    return tf.concat([
        # Shape (num_nodes, num_configs, embedding dim) =(369,10,16)
        tf.stack([graph.node_sets['op']['op_e']] * num_configs, 1), 
        # Shape (num_nodes, num_configs, config feat dim) =(369,10,21 * duplicate factor)
        tf.tile(graph.node_sets['op']['config_feats'], [1, 1, 3]),
        # Shape (num_nodes, num_configs, op feat dim) = (369,10,101)
        tf.stack([graph.node_sets['op']['feats']] * num_configs, 1),
    ], axis=-1) 
     # (369,10,138)
    # Shape (num_nodes, num_configs, 합친 값)
    
    def get_penultimate_output(
      self, pooled: tf.Tensor, graph: tfgnn.GraphTensor,
      num_configs: int) -> tf.Tensor:
        config_feats = graph.node_sets['config']['feats']
        batch_size = graph.node_sets['config'].sizes.shape[0]
        config_feats = tf.reshape(
            config_feats, [batch_size, -1, config_feats.shape[-1]])
        # Shape like config feats
        pooled = tf.stack([pooled] * num_configs, 1)
        pooled = tf.concat([pooled, config_feats], -1)
        return pooled


class ANEEAttentionLayer(layers.Layer):
    def __init__(self, node_dim, edge_dim, hidden_dim, **kwargs):
        super(ANEEAttentionLayer, self).__init__(**kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.Wu = layers.Dense(hidden_dim)
        # This weight vector a is equivalent to the attention mechanism's learnable parameters
        self.a = layers.Dense(1, use_bias=False)
        self.We = layers.Dense(hidden_dim)
        self.Wm = layers.Dense(hidden_dim, use_bias=False)
        self.leaky_relu = layers.LeakyReLU()
        self.softmax = layers.Softmax(axis=1)  # The softmax should be applied across the features

    def call(self, node_features, edge_index, edge_features):
        # First Step: Compute preliminary node features
        h_g_u = self.leaky_relu(self.Wu(node_features))

        # Second Step: Prepare edge features for attention computation
        # Gather the source and target node features for each edge
        edge_source_features = tf.gather(h_g_u, edge_index[:, 1])
        edge_target_features = tf.gather(h_g_u, edge_index[:, 0])
        edge_concat = tf.concat([edge_source_features, edge_target_features], axis=1)
        attention_coefficients = self.a(edge_concat)
        updated_edge_features = self.We(edge_features)  
        edge_feature_interaction = attention_coefficients * updated_edge_features
        e_g_l = self.softmax(edge_feature_interaction)

        # Third Step: Compute the final updated node features
        attention_coefficients2 = self.Wm(e_g_l)  # This is Wm x e_g_l_i
        attention_coefficients2 = tf.nn.softmax(attention_coefficients2, axis=1)  # Apply softmax across features
        neighbor_node_features = tf.gather(node_features, edge_index[:, 0])  # Gather h_g_u_i for each neighbor
        
        # Element-wise multiply the attention coefficients with neighbor node features
        messages = attention_coefficients2 * neighbor_node_features

        # Aggregate messages for each node by summing over all incoming messages from the edges
        # This uses unsorted_segment_sum to sum messages that are targeted to the same node
        aggregated_messages = tf.math.unsorted_segment_sum(
            messages,
            edge_index[:, 1],  # the source node index
            num_segments=tf.shape(node_features)[0]  # The number of segments is the number of nodes
        )

        # Apply LeakyReLU activation function to the aggregated messages
        updated_node_features = self.leaky_relu(aggregated_messages)

        return updated_node_features


class _EarlyJoinWithANEE(_EarlyJoin):
    def __init__(self, num_configs: int, num_ops: int,
                 hidden_dim: int = 64, op_embed_dim: int = 64, **kwargs):
        super(_EarlyJoinWithANEE, self).__init__(num_configs=num_configs, num_ops=num_ops, **kwargs)
        self.anee_attention = ANEEAttentionLayer(node_dim=140,
                                                 edge_dim=24,  # edge_dim should be set according to your model's edge features dimension
                                                 hidden_dim=hidden_dim)
        self.num_configs = num_configs
        self.num_ops = num_ops
        self.op_embed_dim = op_embed_dim
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)

    def attach_config_features_on_op_nodes(self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
        # Existing code for attaching config features
        graph = self.attach_config_features_on_op_nodes_imsi(graph)
        op_features = self.get_op_node_features(graph, num_configs)

        # Add a method or logic to extract and process edge features
        edge_features = self.get_edge_features(graph)  # This method should be defined

        # Apply ANEE's attention mechanism
        attention_output = self.anee_attention(op_features, edge_features)

        # Concatenate the original op_features with the attention output
        combined_features = tf.concat([op_features, attention_output], axis=-1)

        return combined_features

    def get_edge_features(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        # Implement this method based on how your graph represents edge features
        # This is a placeholder and should be replaced with actual implementation
        return tf.zeros((self.num_configs, self.edge_dim))


class _LateJoin(_ConfigFeatureJoiner):
  """Joins module configuration features after applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    del num_configs
    return tf.concat([
        # Shape (num_nodes, embedding dim)
        graph.node_sets['op']['op_e'],
        # Shape (num_nodes, op feat dim)
        graph.node_sets['op']['feats'],
    ], axis=-1)

  def get_penultimate_output(
      self, pooled: tf.Tensor, graph: tfgnn.GraphTensor,
      num_configs: int) -> tf.Tensor:
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    pooled = tf.stack([pooled] * num_configs, 1)
    pooled = tf.concat([pooled, config_feats], -1)
    return pooled
  

class _NewClassWithoutAttentionALL(_EarlyJoinWithANEE):
    def __init__(self, num_configs: int, num_ops: int,
                 hidden_dim: int = 64, op_embed_dim: int = 64, **kwargs):
        super(_EarlyJoinWithANEE, self).__init__(num_configs=num_configs, num_ops=num_ops, **kwargs)
        self.anee_attention = ANEEAttentionLayer(node_dim=140,
                                                 edge_dim=24,  # edge_dim should be set according to your model's edge features dimension
                                                 hidden_dim=hidden_dim)
        self.num_configs = num_configs
        self.num_ops = num_ops
        self.op_embed_dim = op_embed_dim
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)

    def attach_config_features_on_op_nodes(self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
        # Existing code for attaching config features
        graph = self.attach_config_features_on_op_nodes_imsi(graph)
        op_features = self.get_op_node_features(graph, num_configs)

        # Add a method or logic to extract and process edge features
        #edge_features = self.get_edge_features(graph)  # This method should be defined

        # Apply ANEE's attention mechanism
        # attention_output = self.anee_attention(op_features, edge_features)

        # Concatenate the original op_features with the attention output
        #combined_features = tf.concat([op_features, edge_features], axis=-1)

        return op_features

    def get_edge_features(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        # Implement this method based on how your graph represents edge features
        # This is a placeholder and should be replaced with actual implementation
        return tf.zeros((self.num_configs, self.edge_dim))


class _NewClassWithALL(_EarlyJoinWithANEE):
    def __init__(self, num_configs: int, num_ops: int,
                 hidden_dim: int = 64, op_embed_dim: int = 64, **kwargs):
        super(_EarlyJoinWithANEE, self).__init__(num_configs=num_configs, num_ops=num_ops, **kwargs)
        self.anee_attention = ANEEAttentionLayer(node_dim=140,
                                                 edge_dim=24,  # edge_dim should be set according to your model's edge features dimension
                                                 hidden_dim=hidden_dim)
        self.num_configs = num_configs
        self.num_ops = num_ops
        self.op_embed_dim = op_embed_dim
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)

    def attach_config_features_on_op_nodes(self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
        # Existing code for attaching config features
        graph = self.attach_config_features_on_op_nodes_imsi(graph)
        op_features = self.get_op_node_features(graph, num_configs)

        # Add a method or logic to extract and process edge features
        edge_features = self.get_edge_features(graph)  # This method should be defined

        # Apply ANEE's attention mechanism
        attention_output = self.anee_attention(op_features, edge_features)

        # Concatenate the original op_features with the attention output
        combined_features = tf.concat([op_features, attention_output], axis=-1)

        return combined_features

    def get_edge_features(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        # Implement this method based on how your graph represents edge features
        # This is a placeholder and should be replaced with actual implementation
        return tf.zeros((self.num_configs, self.edge_dim))




class LateJoinResGCN(_LateJoin, _ResGCN):
  pass


class EarlyJoinResGCN(_EarlyJoin, _ResGCN):
  pass


class LateJoinSAGE(_LateJoin, _SAGE):
  pass


class EarlyJoinSAGE(_EarlyJoin, _SAGE):
  pass

class EarlyJoinWithAttentionSAGE(_EarlyJoinWithAttention, _SAGE):
  pass

class EarlyJoinWithANEESAGE(_EarlyJoinWithANEE, _SAGE):
  pass


from collections import defaultdict
import numpy as np

def op_jaccard(graph):
    # Extract source and target indices from the 'feed' edge set
    src_indices = graph.edge_sets['feed'].adjacency.source
    tgt_indices = graph.edge_sets['feed'].adjacency.target

    # Calculate the maximum node index
    num_nodes = tf.reduce_max(tf.concat([src_indices, tgt_indices], axis=0)) + 1
    num_nodes= tf.cast(num_nodes, tf.int32)

    # Convert indices to one-hot encoded format and then to boolean
    src_one_hot = tf.cast(tf.one_hot(src_indices, depth=num_nodes), tf.bool)
    tgt_one_hot = tf.cast(tf.one_hot(tgt_indices, depth=num_nodes), tf.bool)

    # Calculate intersection and union for Jaccard coefficient
    # Calculate intersection and union for Jaccard coefficient
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(src_one_hot, tgt_one_hot), tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast(tf.logical_or(src_one_hot, tgt_one_hot), tf.float32), axis=1)


    # Compute Jaccard coefficients
    jaccard_coefficients = tf.math.divide_no_nan(intersection, union)

    # Pool the Jaccard coefficients to the target nodes
    pooled_jac = tfgnn.pool_edges_to_node(graph, 'feed', tfgnn.TARGET, feature_value=jaccard_coefficients) + 1.0

    # Get the 'op' features and apply the transformation
    op_feats = graph.node_sets['op']['feats']
    pooled_jac = tf.reshape(pooled_jac, [-1, 1])
    pooled_jac = tf.cast(pooled_jac, tf.float32)
    result = op_feats * tf.math.exp(pooled_jac)

    # Replace features in the 'op' node set
    op_features = dict(graph.node_sets['op'].features)
    op_features['feats'] = result

    return graph.replace_features(node_sets={'op': op_features})

############################################################
class _JacordSAGE(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GraphSAGE GNN Backbone."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, final_mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 64):
    super().__init__()
    self._num_configs = num_configs
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gnn_layers = []
    for unused_i in range(num_gnns):
      self._gnn_layers.append(_mlp([hidden_dim], hidden_activation))
    self._postnet = _mlp(
        [hidden_dim] * final_mlp_layers + [1], hidden_activation)
    self._activation_fn = getattr(tf.nn, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    return self.forward(graph, self._num_configs)
  

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    #num configs =10
    graph = self._op_embedding(graph) # op_e created 
    # Calculate Jaccard-weighted adjacency matrix
    graph=op_jaccard(graph)
    x = self.get_op_node_features(graph, num_configs) ##(369,10,138)
    # x Shape (num_nodes, num_configs, 합친 값)
    bidirectional_adj = implicit.AdjacencyMultiplier(graph, 'feed') # Shape (num_nodes, num_nodes)
    bidirectional_adj = implicit.Sum(
    bidirectional_adj, bidirectional_adj.transpose())
    for gnn_layer in self._gnn_layers:
      y = bidirectional_adj @ x # Shape (num_nodes, num_configs, 합친 값)
      y = tf.concat([y, x], axis=-1) # Shape (num_nodes, num_configs, 합친 값), residual connection
      y = gnn_layer(y) # Shape (num_nodes, num_configs, hidden_dim)
      y = self._activation_fn(y)
      y = tf.nn.l2_normalize(y, axis=-1)
      x = y # Shape (num_nodes, num_configs, hidden_dim)

    pooled = tfgnn.pool_nodes_to_context(graph, 'op', 'sum', feature_value=x)
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    pooled = self.get_penultimate_output(pooled, graph, num_configs)
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    # _postnet maps across last channel from hidden_dim to 1.

    return tf.squeeze(self._postnet(pooled), -1)


class NewModelWithALLSAGE(_NewClassWithALL, _JacordSAGE):  ### with all : done
  pass

class WithoutJaccardNewModelSAGE(_NewClassWithALL, _SAGE):  ### done
  pass

class WihoutEarlyLatejoinNewModelSAGE(_NewClassWithALL, _JacordSAGE): 
  pass

class WithoutANEENewModelSAGE(_NewClassWithoutAttentionALL, _JacordSAGE): ### 아직
  pass

class WithoutAttentionNewModelSAGE(_NewClassWithoutAttentionALL, _JacordSAGE): ### done
  pass


class MLP(tf.keras.Model):
  """Embeds op codes, averages features across all-nodes, passing thru MLP."""

  def __init__(
      self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
      mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
      hidden_dim: int = 64, reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._reduction = reduction
    layer_dims = [hidden_dim] * mlp_layers
    layer_dims.append(1)
    self._mlp = _mlp(layer_dims, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    op_feats = tf.concat([
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='feats'),
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='op_e'),
    ], axis=-1)

    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    op_feats = tf.stack([op_feats] * num_configs, 1)
    op_feats = tf.concat([op_feats, config_feats], -1)
    return tf.squeeze(self._mlp(op_feats), -1)
