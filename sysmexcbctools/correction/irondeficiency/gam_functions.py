import numpy as np
from pygam import GAM, LinearGAM, s, f, te
from .utils import centralise


def spline_fit(df, col: str, timepoint: str, analyser_ids) -> tuple:
    print(f"Fitting spline for column {col}...")
    center = centralise(df, df.loc[:, col], 3.5)
    # fitting machines individually
    mac1 = center[center[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[0]]
    mac2 = center[center[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[1]]
    return (
        col,
        LinearGAM(
            s(0, n_splines=50)
            + te(
                1,
                2,
                n_splines=30,
                dtype=["numerical", "numerical"],
            )
            # + s(3, n_splines=30, basis="cp")
            + f(3)
            # having more splines for the first term makes sense
            # because there could be sudden discontinuities over
            # the days of the study. Question is how many more and
            # what lambda per term??
            # --> might be sensible to define splines per discontinuity period
        ).fit(
            mac1.loc[
                :,
                [
                    f"FBC_t_d_{timepoint}",
                    f"FBC_tday_h_{timepoint}",
                    f"FBC_sampleage_h_{timepoint}",
                    # f"FBC_tyear_d_{timepoint}",
                    f"FBC_weekday_{timepoint}",
                ],
            ].values,
            mac1.loc[:, col].values,
        ),
        LinearGAM(  # second GAM for machine 2 (kinda dirty fix for the interaction problem)
            s(0, n_splines=50)
            + te(
                1,
                2,
                n_splines=10,
                dtype=["numerical", "numerical"],
            )
            # + s(3, n_splines=30, basis="cp")
            + f(3)
        ).fit(
            mac2.loc[
                :,
                [
                    f"FBC_t_d_{timepoint}",
                    f"FBC_tday_h_{timepoint}",
                    f"FBC_sampleage_h_{timepoint}",
                    # f"FBC_tyear_d_{timepoint}",
                    f"FBC_weekday_{timepoint}",
                ],
            ].values,
            mac2.loc[:, col].values,
        ),
    )


def GAM_predict(df, gam1, gam2, timepoint: str, analyser_ids):
    """function to hide the prediction per machine in the background"""
    pred_features = [
        f"FBC_t_d_{timepoint}",
        f"FBC_tday_h_{timepoint}",
        f"FBC_sampleage_h_{timepoint}",
        # f"FBC_tyear_d_{timepoint}",
        f"FBC_weekday_{timepoint}",
    ]
    res = np.zeros(len(df))
    res[(df[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[0])] = gam1.predict(
        df.loc[
            (df[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[0]),
            pred_features,
        ].values
    )
    res[(df[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[1])] = gam2.predict(
        df.loc[
            (df[f"FBC_analyser_ID_{timepoint}"] == analyser_ids[1]),
            pred_features,
        ].values
    )
    return res
