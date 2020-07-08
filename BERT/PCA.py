
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA


'''
    Incremental PCA class wrapper
'''
class IPCA:
    
    def __init__(self, num_components: int = 150):
        self.num_components = num_components
        self.inc_pca = IncrementalPCA(n_components=self.num_components)
        
        
    def fit(self, x, batch_size : int = 32):
        input_len = len(x)
        iterator = range(0, input_len, batch_size)
        iterator = tqdm(iterator, desc="Processing PCA batches")
        for batch_index in iterator:
            batch_start = batch_index
            batch_end = min(batch_start + batch_size, input_len)
            #print('batch size : ', batch_end-batch_start)
            try:
                self.inc_pca.partial_fit(x[batch_start:batch_end])
            except:
                # batch_size < n_components
                break
        #batches = np.array_split(x, batch_size)
        #for batch in batches:
        #    print('batch size : ', len(batch))
        #     self.inc_pca.partial_fit(batch)
                
    
    def fit_transform(self, x, batch_size : int = 32, with_loss : bool = False):
        self.fit(x, batch_size)
        x_pca = self.transform(x)
        
        if with_loss:
            x_reconstructed = self.reconstruct(x_pca)
            loss = self.compute_loss(x, x_reconstructed)
            return x_pca, x_reconstructed, loss
        
        return x_pca
    
    
    # return "loadings" for vector x
    def transform(self, x):
        x_inc_pca = self.inc_pca.transform(x)
        return x_inc_pca
        
    
    def reconstruct(self, x_pca):
        x = self.inc_pca.inverse_transform(x_pca)
        return x
    
    
    # return the projected vector with reducted dimensionality
    def project(x):
        # first transform, then go back to the initial space
        x_projected = self.inc_pca.inverse_transform(self.transform(x))
        return x_projected
        
    
    def compute_loss(self, x, x_reconstructed):
        return ((x - x_reconstructed) ** 2).mean()
        
        


