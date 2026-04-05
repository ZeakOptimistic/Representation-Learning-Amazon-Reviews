import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import logging

logger = logging.getLogger(__name__)

def split_by_asin(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into train/val/test while preventing leakage.
    Ensures that all reviews for a specific `parent_asin` are placed 
    entirely into train, val, or test.
    """
    if 'parent_asin' not in df.columns:
        logger.warning("parent_asin not found, falling back to random split.")
        # Fallback to random split
        train_val = df.sample(frac=1-test_size, random_state=random_state)
        test = df.drop(train_val.index)
        
        train = train_val.sample(frac=1-(val_size/(1-test_size)), random_state=random_state)
        val = train_val.drop(train.index)
        return train, val, test
        
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['parent_asin']))
    
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    
    # Adjust val_size relative to the remaining train_val set
    adjusted_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val, groups=train_val['parent_asin']))
    
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    
    logger.info(f"Splits -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test
