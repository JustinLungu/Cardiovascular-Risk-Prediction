# Implement missing-value handling, categorical encoding, scaling, and
# feature transformations motivated by exploration and validation results.
# Fit transformation parameters on the training portion only, then apply those
# same parameters and feature order to validation and test data.
# Preserve fitted parameters for reproducible inference; do not modify raw CSVs.


import numpy as np 

# imputers (for now only median and constant)

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
        X = np.array(X, dtype=np.float64, copy=True)
        idx = np.where(np.isnan(X))
        X[idx] = self.med[idx[1]]      # idx[1] columns of nan points 
        return X

    def fit_transform(self, X) : 
        return self.fit(X).transform(X)


class ConstantImp :
    def __init__(self, fill_val : float = -1.0):
        self.fill_val = fill_val

    def fit(self, X):
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        X[np.isnan(X)] = self.fill_val
        return X

    def fit_transform(self, X):
        return self.transform(X)


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
        X = np.array(X, dtype=np.float64, copy=True)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


#OHE

class OHE :
    """One hot encoding on categorical columns, droping the first column to avoid colinearity
    Ignore unknown categories
    Take categories as input """

    def __init__(self, drop_first: bool = True):
        self.drop_first = drop_first
        self.categories = None      #list of list of the values a var can take 

    def fit(self, X):
        cats = []
        for j in range(X.shape[1]) :
            uniq = np.unique(X[:, j])
            if self.drop_first and uniq.size > 1:
                uniq = uniq[1:]
            cats.append(uniq)
        self.categories = cats 
        return self

    def transform(self, X):
        if self.categories is None:
            raise RuntimeError("OHE : call fit first")

        blc = [] 
        for j, cats in enumerate(self.categories):
            val = X[:, j]
            blc.append((val[:, None] == cats[None, :]))
        return np.hstack(blc).astype(np.float64)

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
        self.fitted = None 

    def fit(self, X):
        self.fitted = []
        for name, t, cols in self.transformers:
            t.fit(X[:, cols])
            n_out = t.transform(X[:, cols]).shape[1]
            self.fitted.append((name, t, cols, n_out))
        return self 

    def transform(self, X):
        blc = [t.transform(X[:, cols]) for _, t, cols, _ in self.fitted]
        return np.hstack(blc)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feat_name(self):
        prefix = "X"
        names = []
        for _, t, cols, _ in self.fitted:
            cols = np.flatnonzero(cols)
            if isinstance(t, Pipeline) and isinstance(t.steps[-1][1], OHE):      #t.steps list of tuples (name, tranformer)
                                                                                #suppose ohe is always last in pipeline
                ohe: OHE = t.steps[-1][1]
                for j, cats in zip(cols, ohe.categories):
                    for c in cats:
                        names.append(f"{prefix}{j}_cat{c:g}")
            else:
                for j in cols:
                    names.append(f"{prefix}{j}")
        return names 


    #pipeline 

class Pipeline : 
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        # just like sklearn, fit_transform every step except the last 
        for _, t in self.steps[:-1]:
            X = t.fit_transform(X) 
        self.steps[-1][1].fit(X)
        return self

    def transform(self, X):
        for _, t in self.steps:      
            X = t.transform(X)
        return X

    def fit_transform(self, X):
        for _, t in self.steps[:-1]:
            X = t.fit_transform(X)
        return self.steps[-1][1].fit_transform(X)


###############################################################################


def preproc_MSCOHE(num_inputs, cat_inputs): 
    """Build a pipeline equivalent to the one in sklearn 
    with MSCOHE : Median/scale for num inputs and Constant/OHE for categorical"""

    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", MedianImp()),
            ("scl", StandardScaler()),
        ]), num_inputs),
        ("cat", Pipeline([
            ("imp", ConstantImp(fill_value=-1.0)),
            ("ohe", OHE(drop_first=True)),
        ]), cat_inputs),
    ])





        




    

            
        
             

