# Implement missing-value handling, categorical encoding, scaling, and
# feature transformations motivated by exploration and validation results.
# Fit transformation parameters on the training portion only, then apply those
# same parameters and feature order to validation and test data.
# Preserve fitted parameters for reproducible inference; do not modify raw CSVs.


import numpy as np 

# imputers (for now only median)

class MedianImp :
    """Replace Nan by the median of the column"""
    def __init__(self):
        self.med = None 

    def fit(self, X):
        self.med = np.nanmedian(X, axis = 0) #vector med per column 
        # column full nan forgotten by data expl 
        self.med = np.where(np.isnan(self.med), 0.0, self.med) 
        return self 

    def transform(self, X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = self.med[idx[1]]      # idx[1] columns of nan points 
        return X

    def fit_transform(self, X) : 
        return self.fit(X).tranform(X)


# Scaler 

class StandardScaler :

    def __init__(self):
        self.mean = None
        self.std = None 

    def fit(self, X): 
        self.mean = X.mean(axis = 0)      #mean per column
        self.std = X.std(axis = 0)
        self.std[self.std == 0] = 1.0      #avoid 0-div
        return self 

    def transform(self, X):
        X = X.copy()
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


#OHE

class OHE :
    """One hot encoding on categorical columns, droping the first column to avoid colinearity
    Ignore unknown categories
    Take categories as input """

    def __init__(self, drop_first: bool = True, cat_inputs=None):
        if self.cat_inputs is None :
            raise ValueError("OHE: cat inputs is mandatory")
        self.drop_first = drop_first
        self.cat_inputs = cat_inputs      #list of name of variables 
        self.categories = None      #list of list of the values a var can take 

    def fit(self, X):
        cols = np.flatnonzero(self.cat_inputs)      #change once know the input format
        cats = []
        for j in cols :
            uniq = np.unique(X[:, j])
            if self.drop_first and uniq.size > 1:
                uniq = uniq[1:]
            cats.append(uniq)
        self.categories = cats 
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("call fit first")
        cols = np.flatnonzero(self.cat_inputs)
        blc = [] 
        for j, cats in zip(cols, self.categories):
            val = X[:, j]
            blc.append((val[:, None] == cats[None, :]))
        return np.hstack(blc)

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# column transformer 

class ColumnTransformer :
    """Apply transformer to different groups of columns then concat the result
    Columns not treated by a transformer is dropped 
    
    transformers : list[tuple[str, object, list[int]]]
        ex: [("num", MedianImputer+StandardScaler, [0,1,5]),
             ("cat", ConstantImputer+OneHotEncoder, [2,3,4])]
    """
    def __init__(self, transformers):
        self.transformers = transformers
        self._fitted = None 

    def fit(self, X):
        fitted = []
        used = set()
        for name, t, cols in self.transformers:
            used.update(cols)
            t.fit(X[:, cols])



    

            
        
             

