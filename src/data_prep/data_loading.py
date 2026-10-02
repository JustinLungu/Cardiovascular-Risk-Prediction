# Wrap the unchanged course CSV loader and read feature names from the headers.
# Return features, raw labels, row IDs, and feature names with consistent ordering.
# Check dimensions, feature columns, and feature/label ID alignment.
# Keep labels in {-1, 1} here; convert them when required by a model.
# Loading should not impute, scale, or remove features.
