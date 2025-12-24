"""
GAMCorrector: Scikit-learn-style API for GAM-based covariate correction.
"""

import pickle
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit, logit

# Monkey-patch for pygam compatibility with scipy >= 1.12
# The .A attribute was removed from sparse matrices in scipy 1.12
# This patches pygam to use .toarray() instead
try:
    from scipy.sparse import spmatrix
    if not hasattr(spmatrix, 'A'):
        spmatrix.A = property(lambda self: self.toarray())
except ImportError:
    pass

from pygam import LinearGAM, f, s, te

from .utils import centralise


class GAMCorrector:
    """
    Generalized Additive Model (GAM) based covariate correction.

    This class fits GAMs to correct for spurious covariate effects in data,
    such as sample age, time of day, day of week, or batch effects. It can
    fit separate models for different groups (e.g., different machines).

    Parameters
    ----------
    covariates : list of str
        Names of covariate columns to correct for
    feature_columns : list of str, optional
        Names of feature columns to correct. If None, all numeric columns
        except covariates and group_column will be corrected.
    group_column : str, optional
        Column name for grouping (e.g., 'machine_id'). If provided, separate
        GAMs will be fitted for each group.
    normalize_groups : bool, default=False
        If True and group_column is provided, after applying group-specific
        corrections, normalize groups to have the same mean (remove systematic
        group offsets). This "middles out" the groups to each other.
    reference_group : str, optional
        If normalize_groups=True, normalize all groups to this reference group.
        If None, normalize to the overall mean across all groups.
    transformation : {'log', 'logit', 'none'}, default='none'
        Transformation to apply before fitting:
        - 'log': Natural logarithm (for positive continuous data)
        - 'logit': Logit transform (for proportions/percentages in [0,1])
        - 'none': No transformation
    auto_detect_percentages : bool, default=True
        If True, automatically apply logit transform to columns with 'PCT' or '%'
        in the name and values in [0, 1]
    n_splines : int or dict, default=25
        Number of splines for smooth terms. Can be:
        - int: Same number for all covariates
        - dict: Custom number per covariate (e.g., {'time': 50, 'age': 25})
        Note: Ignored if term_spec is provided.
    term_spec : dict, optional
        Advanced: Specify GAM term types explicitly. If None (default), all covariates
        use smooth spline terms. If provided, must specify term type for each covariate
        or covariate interaction. Examples:
        - {'time': {'type': 's', 'n_splines': 50}} - smooth term
        - {('time', 'age'): {'type': 'te', 'n_splines': 30}} - tensor product interaction
        - {'weekday': {'type': 'f'}} - factor term (categorical)
        - {'time_of_day': {'type': 's', 'n_splines': 25, 'basis': 'cp'}} - cyclic spline
        Term types:
        - 's': Smooth spline (default)
        - 'te': Tensor product for interactions between 2+ continuous covariates
        - 'f': Factor for categorical variables
        Basis types (optional, for 's' terms):
        - 'ps': P-splines (default, if not specified)
        - 'cp': Cyclic penalized splines (for periodic variables)
    centralise_threshold : float, optional
        If provided, filter outliers before fitting using this MAD threshold
    reference_condition : dict, optional
        Condition for calculating reference mean (e.g., {'sample_age': lambda x: x <= 18})
    parallel : bool, default=False
        Whether to fit features in parallel
    n_jobs : int, default=-1
        Number of parallel jobs (-1 means all CPUs)
    verbose : bool, default=True
        Whether to print progress

    Attributes
    ----------
    gam_models_ : dict
        Fitted GAM models for each feature and group
    feature_means_ : dict
        Reference means for each feature
    group_offsets_ : dict
        Group-specific offsets for normalization (if normalize_groups=True)
    transformed_columns_ : list
        Columns that were transformed
    fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    >>> from sysmexcorrect import GAMCorrector
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'WBC': np.random.normal(7.5, 2, 1000) + np.linspace(0, 2, 1000),  # drift over time
    >>>     'HGB': np.random.normal(145, 15, 1000),
    >>>     'time_hours': np.linspace(0, 100, 1000),
    >>>     'sample_age': np.random.uniform(0, 24, 1000)
    >>> })
    >>>
    >>> # Fit and transform
    >>> corrector = GAMCorrector(
    >>>     covariates=['time_hours', 'sample_age'],
    >>>     feature_columns=['WBC', 'HGB']
    >>> )
    >>> df_corrected = corrector.fit_transform(df)
    >>>
    >>> # Save model
    >>> corrector.save('gam_model.pkl')
    >>>
    >>> # Load and apply to new data
    >>> corrector_loaded = GAMCorrector.load('gam_model.pkl')
    >>> df_new_corrected = corrector_loaded.transform(df_new)
    """

    def __init__(
        self,
        covariates: List[str],
        feature_columns: Optional[List[str]] = None,
        group_column: Optional[str] = None,
        normalize_groups: bool = False,
        reference_group: Optional[str] = None,
        transformation: Literal['log', 'logit', 'none'] = 'none',
        auto_detect_percentages: bool = True,
        n_splines: Union[int, Dict[str, int]] = 25,
        term_spec: Optional[Dict] = None,
        centralise_threshold: Optional[float] = None,
        reference_condition: Optional[Dict] = None,
        parallel: bool = False,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        self.covariates = covariates
        self.feature_columns = feature_columns
        self.group_column = group_column
        self.normalize_groups = normalize_groups
        self.reference_group = reference_group
        self.transformation = transformation
        self.auto_detect_percentages = auto_detect_percentages
        self.n_splines = n_splines
        self.term_spec = term_spec
        self.centralise_threshold = centralise_threshold
        self.reference_condition = reference_condition
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Validation
        if self.normalize_groups and self.group_column is None:
            raise ValueError("normalize_groups requires group_column to be specified")

        # Validate term_spec
        if self.term_spec is not None:
            self._validate_term_spec()

        # Fitted attributes
        self.gam_models_ = {}
        self.feature_means_ = {}
        self.group_offsets_ = {}
        self.transformed_columns_ = []
        self.fitted_ = False
        self.groups_ = None

    def _validate_term_spec(self):
        """Validate term_spec structure and content."""
        if not isinstance(self.term_spec, dict):
            raise ValueError("term_spec must be a dictionary")

        # Flatten all covariates mentioned in term_spec
        mentioned_covariates = set()
        for key, spec in self.term_spec.items():
            if isinstance(key, tuple):
                mentioned_covariates.update(key)
            else:
                mentioned_covariates.add(key)

            # Validate spec structure
            if not isinstance(spec, dict):
                raise ValueError(f"term_spec[{key}] must be a dictionary")
            if 'type' not in spec:
                raise ValueError(f"term_spec[{key}] must have a 'type' key")
            if spec['type'] not in ['s', 'te', 'f']:
                raise ValueError(f"term_spec[{key}]['type'] must be 's', 'te', or 'f'")

            # Validate tensor products have multiple covariates
            if spec['type'] == 'te' and not isinstance(key, tuple):
                raise ValueError(f"Tensor product term must use tuple key, got {key}")
            if spec['type'] == 'te' and len(key) < 2:
                raise ValueError(f"Tensor product term must have at least 2 covariates, got {key}")

            # Validate factor terms have single covariate
            if spec['type'] == 'f' and isinstance(key, tuple):
                raise ValueError(f"Factor term must use single covariate, got {key}")

        # Check that all covariates are mentioned in term_spec
        for cov in self.covariates:
            if cov not in mentioned_covariates:
                raise ValueError(
                    f"Covariate '{cov}' not found in term_spec. "
                    f"All covariates must be specified when using term_spec."
                )

    def _get_n_splines(self, covariate: str) -> int:
        """Get number of splines for a covariate."""
        if isinstance(self.n_splines, dict):
            return self.n_splines.get(covariate, 25)
        return self.n_splines

    def _should_transform_logit(self, col: str, values: pd.Series) -> bool:
        """Check if column should be logit transformed."""
        if not self.auto_detect_percentages:
            return False
        # Check if column name contains PCT or %
        has_pct_name = 'PCT' in col or '%' in col
        # Check if values are in [0, 1]
        has_pct_values = (values.min() >= 0) and (values.max() <= 1)
        return has_pct_name and has_pct_values

    def _apply_transformation(self, df: pd.DataFrame, inverse: bool = False) -> pd.DataFrame:
        """
        Apply transformation to feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
        inverse : bool
            If True, apply inverse transformation

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        df = df.copy()

        for col in self.feature_columns:
            if col not in df.columns:
                continue

            # Auto-detect percentage columns
            if self._should_transform_logit(col, df[col]):
                if col not in self.transformed_columns_:
                    self.transformed_columns_.append(col)
                transform_type = 'logit'
            else:
                transform_type = self.transformation

            # Apply transformation
            if not inverse:
                if transform_type == 'log':
                    df[col] = np.log(1e-7 + df[col])
                elif transform_type == 'logit':
                    df[col] = logit(1e-7 + df[col])
            else:
                if transform_type == 'log':
                    df[col] = np.exp(df[col])
                elif transform_type == 'logit':
                    df[col] = expit(df[col])

        return df

    def _build_gam_formula(self) -> LinearGAM:
        """
        Build GAM formula from covariates and optional term_spec.

        Returns
        -------
        LinearGAM
            Unfitted GAM model
        """
        terms = []

        if self.term_spec is None:
            # Default behavior: smooth terms for all covariates
            for i, cov in enumerate(self.covariates):
                n_spl = self._get_n_splines(cov)
                terms.append(s(i, n_splines=n_spl))
        else:
            # Build terms from term_spec
            # Create mapping from covariate name to index
            cov_to_idx = {cov: i for i, cov in enumerate(self.covariates)}

            for key, spec in self.term_spec.items():
                term_type = spec['type']

                if term_type == 's':
                    # Smooth spline term
                    idx = cov_to_idx[key]
                    n_spl = spec.get('n_splines', 25)
                    # Only pass basis if explicitly specified
                    if 'basis' in spec:
                        basis = spec['basis']
                        terms.append(s(idx, n_splines=n_spl, basis=basis))
                    else:
                        terms.append(s(idx, n_splines=n_spl))

                elif term_type == 'te':
                    # Tensor product term
                    indices = [cov_to_idx[cov] for cov in key]
                    n_spl = spec.get('n_splines', 25)
                    terms.append(te(*indices, n_splines=n_spl))

                elif term_type == 'f':
                    # Factor term
                    idx = cov_to_idx[key]
                    terms.append(f(idx))

        # Combine terms
        if len(terms) == 0:
            raise ValueError("At least one term must be specified")

        gam_formula = terms[0]
        for term in terms[1:]:
            gam_formula = gam_formula + term

        return LinearGAM(gam_formula)

    def _fit_single_feature(
        self, df: pd.DataFrame, feature: str, group_value: Optional[str] = None
    ) -> tuple:
        """
        Fit GAM for a single feature (and optionally a single group).

        Parameters
        ----------
        df : pd.DataFrame
            Data to fit
        feature : str
            Feature column name
        group_value : str, optional
            Group identifier

        Returns
        -------
        tuple
            (feature, group_value, fitted_gam, reference_mean)
        """
        # Filter data for this group if applicable
        if group_value is not None:
            fit_df = df[df[self.group_column] == group_value].copy()
        else:
            fit_df = df.copy()

        # Select relevant columns
        cols = [feature] + self.covariates
        fit_df = fit_df[cols].dropna()

        if len(fit_df) < 10:
            if self.verbose:
                print(f"Warning: Only {len(fit_df)} samples for {feature}" +
                      (f" (group={group_value})" if group_value else "") +
                      ", skipping...")
            return (feature, group_value, None, None)

        # Apply centralisation if requested
        if self.centralise_threshold is not None:
            fit_df = centralise(fit_df, fit_df[feature], self.centralise_threshold)

        # Prepare X and y - ensure they're contiguous numpy arrays
        # This helps avoid sparse matrix issues with pygam
        X = np.ascontiguousarray(fit_df[self.covariates].values, dtype=np.float64)
        y = np.ascontiguousarray(fit_df[feature].values, dtype=np.float64)

        # Fit GAM
        try:
            gam = self._build_gam_formula()
            gam.fit(X, y)

            # Calculate reference mean
            if self.reference_condition is not None:
                # Apply reference condition
                mask = np.ones(len(df), dtype=bool)
                for cov, condition_func in self.reference_condition.items():
                    mask &= condition_func(df[cov])
                ref_mean = df.loc[mask, feature].mean()
            else:
                ref_mean = y.mean()

            return (feature, group_value, gam, ref_mean)

        except Exception as e:
            if self.verbose:
                print(f"Error fitting {feature}" +
                      (f" (group={group_value})" if group_value else "") +
                      f": {str(e)}")
            return (feature, group_value, None, None)

    def fit(self, df: pd.DataFrame) -> 'GAMCorrector':
        """
        Fit GAM models for covariate correction.

        Parameters
        ----------
        df : pd.DataFrame
            Training data containing features and covariates

        Returns
        -------
        self
            Fitted GAMCorrector instance

        Raises
        ------
        ValueError
            If required columns are missing
        """
        # Validate inputs
        missing_covs = [c for c in self.covariates if c not in df.columns]
        if missing_covs:
            raise ValueError(f"Missing covariate columns: {missing_covs}")

        # Determine feature columns if not specified
        if self.feature_columns is None:
            exclude = set(self.covariates)
            if self.group_column is not None:
                exclude.add(self.group_column)
            self.feature_columns = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]

        if self.verbose:
            print(f"Fitting GAMs for {len(self.feature_columns)} features...")

        # Apply transformation
        df_transformed = self._apply_transformation(df, inverse=False)

        # Determine groups
        if self.group_column is not None:
            self.groups_ = df[self.group_column].unique().tolist()
        else:
            self.groups_ = [None]

        # Fit models
        if self.parallel:
            from joblib import Parallel, delayed
            from tqdm import tqdm

            from .utils import tqdm_joblib

            tasks = [
                (df_transformed, feat, grp)
                for feat in self.feature_columns
                for grp in self.groups_
            ]

            with tqdm_joblib(tqdm(desc="Fitting GAMs", total=len(tasks))) as pbar:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fit_single_feature)(df, feat, grp)
                    for df, feat, grp in tasks
                )
        else:
            results = []
            for feat in self.feature_columns:
                for grp in self.groups_:
                    result = self._fit_single_feature(df_transformed, feat, grp)
                    results.append(result)

        # Store models
        for feature, group_value, gam, ref_mean in results:
            if gam is not None:
                key = (feature, group_value)
                self.gam_models_[key] = gam
                self.feature_means_[key] = ref_mean

        self.fitted_ = True

        if self.verbose:
            print(f"Successfully fitted {len(self.gam_models_)} GAM models")

        # Calculate group normalization offsets if requested
        if self.normalize_groups and self.group_column is not None:
            if self.verbose:
                print("Calculating group normalization offsets...")

            # Apply current GAMs to get covariate-corrected data
            df_gam_corrected = self._apply_gam_correction(df_transformed)

            # Calculate group means for each feature
            for feature in self.feature_columns:
                if feature not in df_gam_corrected.columns:
                    continue

                group_means = {}
                for group_value in self.groups_:
                    mask = df[self.group_column] == group_value
                    if mask.sum() > 0 and df_gam_corrected.loc[mask, feature].notna().sum() > 0:
                        group_means[group_value] = df_gam_corrected.loc[mask, feature].mean()

                # Determine reference value
                if self.reference_group is not None and self.reference_group in group_means:
                    reference_value = group_means[self.reference_group]
                else:
                    # Use overall mean across all groups
                    reference_value = np.mean(list(group_means.values()))

                # Calculate offsets (shift each group to reference)
                for group_value, group_mean in group_means.items():
                    key = (feature, group_value)
                    self.group_offsets_[key] = reference_value - group_mean

            if self.verbose:
                print(f"Calculated normalization offsets for {len(self.group_offsets_)} feature-group combinations")

        return self

    def _apply_gam_correction(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        """
        Apply GAM correction to transformed data (internal helper).

        Parameters
        ----------
        df_transformed : pd.DataFrame
            Data that has already been transformed (log/logit)

        Returns
        -------
        pd.DataFrame
            GAM-corrected data (still in transformed space)
        """
        df_corrected = df_transformed.copy()

        # Apply GAM correction
        for feature in self.feature_columns:
            if feature not in df_corrected.columns:
                continue

            for group_value in self.groups_:
                key = (feature, group_value)
                if key not in self.gam_models_:
                    continue

                gam = self.gam_models_[key]
                ref_mean = self.feature_means_[key]

                # Select rows to correct
                if group_value is not None:
                    mask = (df_corrected[self.group_column] == group_value)
                else:
                    mask = np.ones(len(df_corrected), dtype=bool)

                # Also check that covariates are not missing
                for cov in self.covariates:
                    mask &= df_corrected[cov].notna()

                mask &= df_corrected[feature].notna()

                if mask.sum() == 0:
                    continue

                # Predict and correct
                X = np.ascontiguousarray(
                    df_corrected.loc[mask, self.covariates].values,
                    dtype=np.float64
                )
                predictions = gam.predict(X)
                df_corrected.loc[mask, feature] = (
                    df_corrected.loc[mask, feature] - predictions + ref_mean
                )

        return df_corrected

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply GAM-based correction to data.

        If normalize_groups=True, also applies group normalization after
        GAM correction to align groups to each other.

        Parameters
        ----------
        df : pd.DataFrame
            Data to correct

        Returns
        -------
        pd.DataFrame
            Corrected data

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not self.fitted_:
            raise ValueError("GAMCorrector must be fitted before transform()")

        df_corrected = df.copy()

        # Apply transformation
        df_corrected = self._apply_transformation(df_corrected, inverse=False)

        # Apply GAM correction
        df_corrected = self._apply_gam_correction(df_corrected)

        # Apply group normalization if requested
        if self.normalize_groups and self.group_column is not None:
            for feature in self.feature_columns:
                if feature not in df_corrected.columns:
                    continue

                for group_value in self.groups_:
                    key = (feature, group_value)
                    if key not in self.group_offsets_:
                        continue

                    offset = self.group_offsets_[key]

                    # Select rows for this group
                    mask = (df[self.group_column] == group_value) & df_corrected[feature].notna()

                    if mask.sum() == 0:
                        continue

                    # Apply offset
                    df_corrected.loc[mask, feature] += offset

        # Apply inverse transformation
        df_corrected = self._apply_transformation(df_corrected, inverse=True)

        return df_corrected

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit GAM models and apply correction in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Data to fit and correct

        Returns
        -------
        pd.DataFrame
            Corrected data
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str):
        """
        Save fitted GAMCorrector to disk.

        Note: Lambda functions in `reference_condition` cannot be pickled and will be
        excluded from the saved model. The model can still be used for prediction,
        but cannot be re-fitted with the original reference condition.

        Parameters
        ----------
        filepath : str
            Path to save the model (should end in .pkl)
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted GAMCorrector")

        # Temporarily store unpicklable attributes
        unpicklable_attrs = {}
        warnings_issued = []

        # Check if reference_condition contains lambda functions
        if self.reference_condition is not None:
            has_lambda = False
            for key, value in self.reference_condition.items():
                if callable(value) and value.__name__ == '<lambda>':
                    has_lambda = True
                    break

            if has_lambda:
                unpicklable_attrs['reference_condition'] = self.reference_condition
                self.reference_condition = None
                warnings_issued.append("reference_condition (contains lambda functions)")

        # Issue warning if anything was excluded
        if warnings_issued:
            import warnings as warn_module
            warn_module.warn(
                f"The following attributes contain lambda functions and cannot be pickled: "
                f"{', '.join(warnings_issued)}. They have been excluded from the saved model. "
                f"The model can still be used for prediction, but cannot be re-fitted with "
                f"these attributes.",
                UserWarning
            )

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        finally:
            # Restore unpicklable attributes
            for attr_name, attr_value in unpicklable_attrs.items():
                setattr(self, attr_name, attr_value)

    @classmethod
    def load(cls, filepath: str) -> 'GAMCorrector':
        """
        Load fitted GAMCorrector from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model

        Returns
        -------
        GAMCorrector
            Loaded model
        """
        with open(filepath, 'rb') as f:
            corrector = pickle.load(f)

        if not isinstance(corrector, cls):
            raise ValueError("Loaded object is not a GAMCorrector")

        return corrector

    def get_partial_dependence(self, feature: str, covariate: str,
                                group_value: Optional[str] = None) -> pd.DataFrame:
        """
        Get partial dependence of a feature on a covariate.

        Parameters
        ----------
        feature : str
            Feature name
        covariate : str
            Covariate name
        group_value : str, optional
            Group identifier

        Returns
        -------
        pd.DataFrame
            Partial dependence data
        """
        if not self.fitted_:
            raise ValueError("GAMCorrector must be fitted before getting partial dependence")

        key = (feature, group_value)
        if key not in self.gam_models_:
            raise ValueError(f"No model found for feature='{feature}', group='{group_value}'")

        gam = self.gam_models_[key]
        cov_idx = self.covariates.index(covariate)

        # Generate grid for this covariate
        XX = gam.generate_X_grid(term=cov_idx)

        # Get partial dependence
        pdep, confi = gam.partial_dependence(term=cov_idx, X=XX, width=0.95)

        return pd.DataFrame({
            'covariate_value': XX[:, cov_idx],
            'partial_dependence': pdep,
            'lower_ci': confi[:, 0],
            'upper_ci': confi[:, 1]
        })
