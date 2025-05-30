# # A simple implementation of matrix factorization for collaborative filtering expressed as a Keras Sequential model

# # Keras uses TensorFlow tensor library as the backend system to do the heavy compiling

# import numpy as np
# from keras.layers import Embedding, Reshape, Merge
# from keras.models import Sequential

# class CFModel(Sequential):

#     # The constructor for the class
#     def __init__(self, n_users, m_items, k_factors, **kwargs):
#         # P is the embedding layer that creates an User by latent factors matrix.
#         # If the intput is a user_id, P returns the latent factor vector for that user.
#         P = Sequential()
#         P.add(Embedding(n_users, k_factors, input_length=1))
#         P.add(Reshape((k_factors,)))

#         # Q is the embedding layer that creates a Movie by latent factors matrix.
#         # If the input is a movie_id, Q returns the latent factor vector for that movie.
#         Q = Sequential()
#         Q.add(Embedding(m_items, k_factors, input_length=1))
#         Q.add(Reshape((k_factors,)))

#         super(CFModel, self).__init__(**kwargs)
        
#         # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
#         self.add(Merge([P, Q], mode='dot', dot_axes=1))

#     # The rate function to predict user's rating of unrated items
#     def rate(self, user_id, item_id):
#         return self.predict([np.array([user_id]), np.array([item_id])])[0][0]


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Reshape, Dot, Input
from tensorflow.keras.models import Model

class CFModel(Model):
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        super(CFModel, self).__init__(**kwargs)
        
        # 保存维度信息
        self.n_users = n_users
        self.m_items = m_items
        self.k_factors = k_factors
        
        # 定义层
        self.user_embedding = Embedding(n_users, k_factors, input_length=1)
        self.item_embedding = Embedding(m_items, k_factors, input_length=1)
        self.user_reshape = Reshape((k_factors,))
        self.item_reshape = Reshape((k_factors,))
        self.dot_product = Dot(axes=1)
        
    def call(self, inputs):
        user_input, item_input = inputs
        user_embedded = self.user_reshape(self.user_embedding(user_input))
        item_embedded = self.item_reshape(self.item_embedding(item_input))
        return self.dot_product([user_embedded, item_embedded])
    
    def get_config(self):
        config = super(CFModel, self).get_config()
        config.update({
            'n_users': self.n_users,
            'm_items': self.m_items,
            'k_factors': self.k_factors
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0]