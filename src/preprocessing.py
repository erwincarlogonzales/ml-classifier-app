from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# Create preprocessing pipeline
def create_preprocessing_pipeline():
    
    # Select numeric and categorical columns
    num_cols = make_column_selector(dtype_include = 'number')
    cat_cols = make_column_selector(dtype_include = 'object')
    
    # Instantiate the transformers
    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    
    # Create pipeline
    num_pipe = Pipeline([
        ('imputer', knn_imputer),
        ('scaler', scaler),
    ])
    
    cat_pipe = Pipeline([
        ('encoder', ohe)
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols),
    ], remainder='drop')
    
    return preprocessor

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create sampler pipeline
def make_sampler_pipeline(sampler):
    return ImbPipeline([
        ('sampler', sampler)
    ])
    
# Preprocess and rebalance data
def preprocess_and_rebalance_data(preprocessor, X_train, X_test, y_train):
    
    # Transform training data into the fitted transformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Create sampling pipeline
    sampler = make_sampler_pipeline(SMOTE(random_state=42))
    
    # Balance training data
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_transformed, y_train)
    
    return X_train_balanced, X_test_transformed, y_train_balanced