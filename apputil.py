import pandas as pd

class GroupEstimate:
    """A class for grouping and estimating mean or median values."""
    
    def __init__(self, estimate):
        if estimate not in ["mean", "median"]:
            raise ValueError("Estimate must be either 'mean' or 'median'.")
        self.estimate = estimate
        self.group_estimates = None

    def fit(self, X, y):
        """Fit the model with the training data."""
        
        # Combine X and y into a DataFrame
        data = pd.DataFrame({"X": X, "y": y})

        # Check for missing values
        if data.isnull().any().any():
            raise ValueError("Input data contains missing values.")

        # Group by the categorical column X
        grouped = data.groupby("X")

        # Calculate the mean or median for each group
        if self.estimate == "mean":
            self.group_estimates = grouped["y"].mean()
        elif self.estimate == "median":
            self.group_estimates = grouped["y"].median()

    def predict(self, X):
        """Predict the estimates for new data."""
        
        # Ensure the model has been fit
        if self.group_estimates is None:
            raise ValueError("The model must be fit before calling predict.")
        
        # Convert X into a pandas Series
        X = pd.Series(X)

        # Map X to the group estimates
        predictions = X.map(self.group_estimates)
        
        # Find missing categories
        missing_groups = predictions.isna().sum()

        # Print a message if there are missing categories
        if missing_groups > 0:
            print(f"{missing_groups} observation(s) correspond to missing groups in the training data.")

        # Return predictions as a NumPy array
        return predictions.to_numpy()