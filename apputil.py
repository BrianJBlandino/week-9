import pandas as pd


class GroupEstimate:
    """Creating a class that accepts an estimate argument
    that can be either 'mean' or 'median'."""
    
    def __init__(self, estimate):
        """Creating a function that sets the standards for
        the variables and initializes the class and assigns
        the instance variable."""
        
        if estimate not in ["mean", "median"]:
            raise ValueError("Estimate must be either 'mean' or 'median'.")
        self.estimate = estimate
        self.group_estimates = None

    def fit(self, X, y):
        """Creating a function that takes in a pandas Dataframe
        of categorical data 'X' and a 1-D array 'y'."""
        
        # Combining X and y into a single DataFrame
        data = pd.DataFrame({"X": X, "y": y})

        # Checking for missing values
        if data.isnull().any().any():
            raise ValueError("Input data contains missing values.")
        
        # Grouping by the categorical column X
        grouped = data.groupby("X")

        # Calculating the mean or median for each group based on the estimate argument
        if self.estimate == "mean":
            self.group_estimates = grouped["y"].mean()
        elif self.estimate == "median":
            self.group_estimates = grouped["y"].median()
            
    def predict(self, X):
        """Creating a function that takes in an array of observations
        (or a dataframe) corresponding to the columns in 'X', then
        determines which group they fall into, and returns the
        corresponding estimates for 'y'."""
        
        # Ensuring the model has been fit
        if self.group_estimates is None:
            raise ValueError("The model must be fit before calling predict.")

        # Converting X into a pandas Series for compatibility
        X = pd.Series(X)

        # Mapping X to the group estimates
        predictions = X.map(self.group_estimates)
        
        # Find missing categories
        missing_groups = predictions.isna().sum()

        # Print a message if there are missing categories
        if missing_groups > 0:
            print(f"{missing_groups} observation(s) correspond to missing groups in the training data.")

        # Returning the 'predictions' as a NumPy array
        return predictions.to_numpy()