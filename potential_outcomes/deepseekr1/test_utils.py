import pandas as pd
import numpy as np
from utils import load_data
import os
import tempfile


def test_load_data_nan_handling():
    """Test NaN handling in load_data function"""
    # Create a test DataFrame with NaN values
    data = {
        'numeric1': [1, 2, np.nan, 4],
        'numeric2': [np.nan, 2, 3, 4],
        'categorical1': ['a', 'b', np.nan, 'c'],
        'categorical2': [np.nan, 'x', 'y', 'z'],
        'boolean': [1, 0, 1, np.nan]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        test_file = f.name
    
    # Load and preprocess
    processed_df = load_data(test_file)
    
    # Check numeric columns
    assert processed_df['numeric1'].isnull().sum() == 0
    assert processed_df['numeric2'].isnull().sum() == 0
    assert processed_df['numeric1'].iloc[2] == 2.5  # median of [1,2,4] is 2
    assert processed_df['numeric2'].iloc[0] == 3.0  # median of [2,3,4] is 3
    
    # Check categorical columns
    assert processed_df['categorical1'].isnull().sum() == 0
    assert processed_df['categorical2'].isnull().sum() == 0
    assert processed_df['categorical1'].iloc[2] == 'a'  # mode might be 'a' (or any of a,b,c)
    
    # Check boolean column
    assert processed_df['boolean'].isnull().sum() == 0
    assert processed_df['boolean'].iloc[3] == 1  # mode of [1,0,1] is 1
    
    # Clean up
    os.unlink(test_file)
    print("test_load_data_nan_handling passed!")


if __name__ == "__main__":
    test_load_data_nan_handling()
