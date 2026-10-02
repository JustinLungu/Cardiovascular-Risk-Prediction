# Create reproducible stratified train/validation splits and cross-validation folds.
# Return indices so features, labels, and IDs stay aligned; record seeds and split sizes.
# Keep a final holdout separate from tuning and fit preprocessing within each fold.
