"""
IQR Transformer
===============
"""
import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# --------------------------------------------------------------------------
#                       Inter-Quantile Range filter
# --------------------------------------------------------------------------
class IQRTransformer(BaseEstimator, TransformerMixin):
    """Description...

    .. note: Does it make to do it for different classes?
    .. note: __repr__ using scikits methods?
    """

    def __init__(self, iqrrange=[25, 75], coefficient=1.5):
        """The constructor"""
        self.iqrrange = iqrrange
        self.coefficient = coefficient
        self.lower_coefs_ = None
        self.upper_coefs_ = None

    def __repr__(self):
        """The representation"""
        return "IQRFilter(iqrrange=%s, coefficient=%s)" % \
               (self.iqrrange, self.coefficient)

    # --------------------------
    # helper methods
    # --------------------------
    def _fit(self, X, y=None):
        """This method computes the lower and upper percentiles
        """
        # Compute lower and uper quartiles
        lower_quartiles, upper_quartiles = \
            np.nanpercentile(X, self.iqrrange, axis=0)

        # Compute the interquantile range
        iqrs = (upper_quartiles - lower_quartiles) * self.coefficient

        # Set parameters
        return lower_quartiles - iqrs, upper_quartiles + iqrs

    def _transform(self, X, y=None):
        """This method filters single category.

        Parameters
        ----------
        X :

        Returns
        -------
        np.ndarray
        """
        # Copy X
        F = np.copy(X)

        # Indexes
        is_lower = F < self.lower_coefs_[0, :]
        is_upper = F > self.upper_coefs_[0, :]

        # Filter
        F[is_lower | is_upper] = np.nan

        # Return
        return F

    # --------------------------
    # main methods
    # --------------------------
    def fit(self, X, y=None):
        """This method fits single category.

        Parameters
        ----------
        X :

        Returns
        -------
        IQRFIlter instance
        """
        # Create the array coefficients
        self.lower_coefs_, self.upper_coefs_ = self._fit(X)

        # Format to array
        self.lower_coefs_ = self.lower_coefs_.reshape(1, -1)
        self.upper_coefs_ = self.upper_coefs_.reshape(1, -1)

        # Return
        return self

    def transform(self, X, y=None):
        """ This method...
        """
        # The object has not been previously fitted
        if self.lower_coefs_ is None or self.upper_coefs_ is None:
            raise TypeError("The instance IQRFilter has not been fitted.")

        # Return
        return self._transform(X, y)



class DataFrameMixin:
    """"""

    def __init__(self, include=[], exclude=[]):
        """"""
        self.include = include
        self.exclude = exclude

    def get_include(self, X):
        """"""
       # if none of them (include all)
       # if include then include
       # if exclude then exclude
       # if both of them include - exclude.
        #if not self.include and not self.exclude:
        pass


class AggregateTransformer(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self, aggmap={}, include=[], by=None, labels=[]):
        """The constructor"""
        self.by = by
        self.aggmap = {}
        self.aggmap.update(aggmap)
        #self.exclude = [self.by, self.date]
        self.include = include
        self.labels = labels

    def __repr__(self):
        """The representation"""
        return "AggregateTransformer(by=%s, aggmap=%s)" % \
               (self.by, self.aggmap)

    def get_X(self):
        """"""
        pass

    def get_y(self):
        """"""
        pass

    def fit(self, X, y=None):
        """Fit the transformer.

        .. note: If a row contains number representing categories
                 (e.g. [1,2,3]) you should define those categories
                 as non numeric strings. Otherwise, the max function
                 will be associated to them.
        """
        # Convert to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).convert_dtypes()

        # Add missing aggregation functions
        for c in X.columns:
            if c not in self.aggmap:
                if not c in self.include:
                    continue
                try:
                    pd.to_numeric(X[c])
                    self.aggmap[c] = 'max'
                except:
                    self.aggmap[c] = 'first'

        # Return
        return self

    def transform(self, X, y=None):
        """Apply transformation.

        .. note: The Transformer needs self.by. When not included
                 the last line of this method is called (and it
                 raises an error).
        """
        # Check if all columns in X were fitted previously.

        if not isinstance(X, pd.DataFrame):
            raise Exception("""The input parameter <X> must be a 
                DataFrame containing the following columns: %s"""
                % self.aggmap.keys())

        # Transform
        aux = X.copy(deep=True)
        aux = aux.groupby(by=self.by)
        aux = aux[self.include].agg(self.aggmap)
        aux.columns = ['_'.join(col).strip()
            for col in aux.columns.values]

        # Return
        return aux.reset_index()

        if self.by is not None:
            return X.groupby(by=self.by).agg(self.aggmap)
        X[self.by] = 'unknown'
        return X.groupby(by=self.by).agg(self.aggmap)


class DeltaTransformer(BaseEstimator, TransformerMixin):
    """Delta between the current and previous element.

    Computes the delta change from the immediately previous row by
    default. This is useful in comparing the delta changes (absolute
    or percentages) in a time series of elements.

    .. todo: Add include / exclude.
       if none of them (include all)
       if include then include
       if exclude then exclude
       if both of them include - exclude.


    Parameters
    ----------
    by: str, list (default None)
        Column names to group by before applying the function. For
        example, this could be used to define an individual (e.g.
        patient)

    date: str (default None)
        Column containing the datetime information. This must
        argument must be passed if re-sampling wants to be done.
        Otherwise, the function will be applied to consecutive elements
        even if there are dates in between which do not appear in the
        data.

    resample: boolean
        Whether to re-sample the data. It requires date to be specified.

    resample_params: dict-like

    periods: list
        List of integers with periods.

    method: string
        Whether to compute the percentage.

    keep: boolean
        Whether to keep the original features.

    resample_params: dict-like

    function_params: dict-like

    """
    def __init__(self, by=None, date=None, periods=[1],
            method='diff', keep=True, include=[],
            resample_params={'rule':'1D'},
            function_params={}):
        """The constructor"""
        self.by = by
        self.date = date
        self.periods = periods
        self.method = method
        self.keep = keep
        self.resample_params = resample_params
        self.function_params = function_params
        self.include = include

    def __repr__(self):
        """The representation"""
        return "DeltaTransformer(by=%s, periods=%s, method=%s)" % \
               (self.by, self.periods, self.method)

    def fit(self, X, y=None):
        """Fit the transformer."""

        #self._validate_data(X, accept_sparse=False, reset=False)

        # Convert to DataFrame
        #if not isinstance(X, pd.DataFrame):
        #    X = pd.DataFrame(X).convert_dtypes()

        if not self.include:
            self.include = X.columns.tolist()
            for e in [self.by, self.date]:
                if not e in self.include:
                    continue
                self.include.remove(e)

        # Return
        return self

    def transform(self, X, y=None):
        """Apply transformation.

        .. note: Do i NEed to ensure sorting?

        """

        # Checks (what about columns?)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).convert_dtypes()

        # Cast date column to datetime
        if self.date is not None:
            X[self.date] = pd.to_datetime(X[self.date])

        #if self.by is not None:
        #    return X.groupby(by=self.by).agg(self.aggmap)
        #X['study_no'] = 'unknown'

        def delta(x, periods=1, method='diff', **kwargs):
            """Computes delta (diff between days)

            Parameters
            ----------
            x: pd.dataFrame
                The DataFrame
            periods: int
                The periods.

            Returns
            -------
            """
            if not callable(method):
                if method == 'diff':
                    aux = x.diff(periods=periods, **kwargs)
                elif method == 'pct_change':
                    aux = x.pct_change(periods=periods, **kwargs)

            aux = aux.add_suffix('_d%s' % periods)
            return aux

        def resample_01(df, ffill=True, **kwargs):
            return df.droplevel(0) \
                .resample(**kwargs) \
                .asfreq() \
                # .ffill() # filling missing!

        # Set by
        if self.by:
            X = X.set_index(self.by)

        # Re-sample
        #What if no patient and only date.
        if self.date:
            X = X.set_index([self.date], append=True) \
                .groupby(level=0) \
                .apply(resample_01, **self.resample_params)

        # Compute delta changes
        aux = pd.DataFrame()
        for p in self.periods:
            # Compute delta
            df_ = X.groupby(self.by)[self.include] \
                .apply(delta, periods=p, method=self.method)
            aux = pd.concat([aux, df_], axis=1)

        # Keep original columns
        if self.keep:
            aux = pd.concat([X, aux], axis=1)

        # Return
        return aux.reset_index()



if __name__ == '__main__':

    # Import
    import numpy as np
    import warnings

    # Import specific
    from sklearn.datasets import make_classification

    # ------------------------------------
    # basic configuration
    # ------------------------------------
    # Ignore all the warnings
    warnings.simplefilter('ignore')

    # Set print options
    np.set_printoptions(precision=2)

    # ------------------------------------
    # create data
    # ------------------------------------
    # Create feature data
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 1, 2, 3],
                     [1, 2, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 4],
                     [3, 6, 3, 4],
                     [2, 2, -55, 55]], np.float64)

    # What if data has other non-numbers.

    # Create categories
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # --------------
    # IQR filtering
    # --------------
    # Create filter object
    iqr = IQRTransformer(iqrrange=[25, 75], coefficient=1.5)

    # .. note: There is no need to pass y, it is just for
    #          compatibility with other transformers.

    # Fit and transform
    print("\n\n" + "-"*50)
    print("IQRTransformer")
    print("-"*50)
    print(iqr)
    print("\nOriginal:")
    print(data)
    print("\nTransformed:")
    print(iqr.fit_transform(data))
    print("\nTransformed:")
    print(iqr.fit_transform(data, y))

    # --------------
    # Aggregation
    # --------------
    """
    # Create aggmap
    agg = AggregateTransformer()

    # Fit and transform
    print("\n\nAggregateTransformer")
    print("-"*50)
    print("\nOriginal:")
    print(data)
    print("\nTransformed:")
    print(agg.fit_transform(data))
    """

    # Create feature data
    # Note when wrapping around np.array for
    # some reason the convert dtypes does not
    # work.
    data = [['a', '2022-01-01', 1, 2, 3, 4, True],
            ['a', '2022-01-02', 5, 6, 7, 8, True],
            ['a', '2022-01-04', 9, 1, 2, 3, True],
            ['b', '2022-01-01', 1, 2, 3, 4, True],
            ['b', '2022-01-03', 1, 2, 3, 3, False],
            ['b', '2022-01-02', 1, 7, 3, 4, False],
            ['c', '2022-01-01', 1, 2, 3, 3, False],
            ['c', '2022-01-02', 1, 7, 3, 4, False],
            ['c', '2022-01-03', 1, 3, 1, 4, False]]

    # What if two equal dates?!
    # What if missing days?

    # Assign names to columns
    data = pd.DataFrame(data,
        columns=['patient', 'date', 'bp', 'hr', 'hrv', 'pulse', 'y'])

    # Create aggregation map
    aggmap = {
        'bp': 'mean',
        'hr': ['min', 'max', 'median'],
        'y': ['last']
    }
    agg = AggregateTransformer(aggmap=aggmap,
        include=['bp', 'hr', 'y'],
        labels=['y'],
        by='patient')


    # Fit and transform
    print("\n\n" + "-"*50)
    print("AggregateTransformer")
    print("-"*50 + '\n')
    print(agg)
    print("\nOriginal:")
    print(data)
    print("\nTransformed:")
    print(agg.fit_transform(data))

    """
    # Further testing...
    #agg.fit_transform(data.to_numpy()) # numpy input
    agg.fit_transform(data[['patient', 'hr']])
    agg_t1 = AggregateTransformer(aggmap={}) #.fit_transform(data)
    agg_t1 = agg_t1.fit_transform(data)
    print(agg_t1)
    """

    # -----------------
    # Delta Transformer
    # -----------------
    # Create filter object
    delta = DeltaTransformer(by='patient',
        date='date', include=[],
        periods=[1,2], method='diff',
        resample_params={'rule': '1D'},
        function_params={'fill_method': 'ffill'})

    # Fit and transform
    print("\n\n" + "-"*50)
    print("DeltaTransformer")
    print("-"*50 + '\n')
    print(delta)
    print("\nOriginal:")
    print(data)
    print("\nTransformed:")
    print(delta.fit_transform(data))

    # At the moment, by and date are required. See if it would
    # be interesting to allow this transformer to work when
    # one or both of this parameters are not defined.

    #delta_t1 = DeltaTransformer(periods=[1,2])
    delta_t2 = DeltaTransformer(by='patient', periods=[1, 2])

    #print("\nTransformed (T1):")
    #print(delta_t1.fit_transform(data))
    #print("\nTransformed (T2):")
    #print(delta_t2.fit_transform(data))
