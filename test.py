from tests import ( test_random_choice,
                    test_vectors)

if __name__ == "__main__":
#    test_random_choice.test()
    test_vectors.test()
    
#%%


#%%













#%%

from mowgly.text.specs import CharColumnSpec
#%%
import pandas as pd
#%%
corpus = pd.read_pickle('/home/jonatan/Dropbox/FairyText/data/dawiki.pkl')
#%%
#corpus = corpus.iloc[:10000]
#%%
import numpy as np

specs = [CharColumnSpec(column_name='text', 
                        language='danish', 
                        lower=True, 
                        min_len=1, 
                        max_len=2000,
                        sample_len=200,
                        fixed_length_subsample=True, 
                        uniform_start=False, 
                        dtype=np.int16),
        CharColumnSpec(column_name='orig_title', 
                        language='danish', 
                        lower=True, 
                        min_len=1, 
                        max_len=100, 
                        fixed_length_subsample=False, 
                        uniform_start=False, 
                        dtype=np.int16)
        ]
#%%
from mowgly.text import TextColumnStore
store = TextColumnStore(corpus, specs, n_jobs=4)
#%%
import numpy as np
indices = np.random.randint(0,store.size,256).astype(np.int32)
#%%
#%timeit out = store[indices]