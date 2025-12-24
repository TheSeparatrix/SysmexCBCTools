import importlib

from irondeficiency._feature_lists_decrypted import (
    hl_features,
    uncorr_features,
)

pretty_feature_names = {
    "non_blood": "Non-blood",
    "HGB_bl": "Haemoglobin",
    "standard_FBC_bl": "Standard FBC",
    "HD_FBC_bl": "HD-FBC",
    "three_iron_bl": "HGB, MCV, MCH",
    "standard_FBC_24m": "Standard FBC",
    "standard_FBC_bl_24m": "Standard FBC",
    "HD_FBC_24m": "HD-FBC",
    "three_iron_24m": "HGB, MCV, MCH",
    "HD_FBC_bl_24m": "HD-FBC (2 timepoints)",
    "HD_FBC_bl_24m_ferritin": "HD-FBC w/ historic\nHD-FBC+ferritin",
    "standard_FBC_bl_24m_ferritin": "Standard FBC w/ historic\nFBC+ferritin",
    "standard_FBC_bl_24m_ferritin_trialarm": "Standard FBC w/ historic\nFBC+ferritin\nand trial arm severity",
    "standard_FBC_bl_24m_delta_ferritin_trialarm": "Standard FBC w/ historic\nFBC+ferritin\nand trial arm severity",
    "HD_FBC_bl_24m_ferritin_trialarm": "HD-FBC w/ historic\nHD-FBC+ferritin\nand trial arm severity",
    "standard_FBC_trialarm_deferral": "Standard FBC\nand trial arm severity",
    "HD_FBC_trialarm_deferral": "HD-FBC\nand trial arm severity",
    "HD_FBC_ferritin_trialarm_deferral": "HD-FBC w/ ferritin\nand trial arm severity",
    "HD_FBC_sex": "HD-FBC",
    "standard_FBC_alltimepoints": "Standard FBC",
    "three_iron_alltimepoints": "HGB, MCV, MCH",
    "HD_FBC_alltimepoints": "HD-FBC",
    "HD_FBC_uncorr_alltimepoints": "HD-FBC",
    "HD_FBC_sex_alltimepoints": "HD-FBC",
    "HD_FBC_bl_ferritin_trialarm": "HD-FBC+Ferr+TA\n(BL)",
    "HD_FBC_bl_24m_ferritin_trialarm": "HD-FBC+Ferr+TA\n(BL+24M)",
    "HD_FBC_noRET_noPLTF_noIP": "HD-FBC\n(no RET, PLT-F, messages)",
    "HD_FBC_noRET_noPLTF": "HD-FBC\n(no RET, PLT-F)",
    "HD_FBC_noRET": "HD-FBC\n(no RET)",
    "HD_FBC_noPLTF": "HD-FBC\n(no PLT-F)",
    "HD_FBC_noIP": "HD-FBC\n(no messages)",
    "HD_FBC_noRET_noIP": "HD-FBC\n(no RET, messages)",
    "HD_FBC_noPLTF_noIP": "HD-FBC\n(no PLT-F, messages)",
    "HD_FBC_noGaussian": "HD-FBC\n(no Gaussian)",
    "HD-FBC_noGaussian_noRET": "HD-FBC\n(no Gaussian, RET)",
    "HD-FBC_noGaussian_noRET_noPLTF": "HD-FBC\n(no Gaussian, RET, PLT-F)",
    "HD-FBC_noGaussian_noRET_noPLTF_noIP": "HD-FBC\n(no Gaussian, RET, PLT-F, messages)",
    "HD-FBC_noGaussian_noIP": "HD-FBC\n(no Gaussian, messages)",
    "HD-FBC_noGaussian_noPLTF": "HD-FBC\n(no Gaussian, PLT-F)",
    "HD-FBC_noGaussian_noRET_noIP": "HD-FBC\n(no Gaussian, RET, messages)",
    "HD-FBC_noGaussian_noPLTF_noIP": "HD-FBC\n(no Gaussian, PLT-F, messages)",
}

# get features
feature_lists = importlib.import_module(f"irondeficiency._feature_lists_decrypted")
standard_fbc_features_bl = getattr(feature_lists, "hl_features_bl")
hd_fbc_features_bl = getattr(feature_lists, "baseline_blood_feats")
three_iron_features_bl = ["HGB_g_L_bl", "MCV_fL_bl", "MCH_pg_bl"]
standard_fbc_features_24m = getattr(feature_lists, "hl_features_24m")
hd_fbc_features_24m = getattr(feature_lists, "twoyear_blood_feats")
three_iron_features_24m = ["HGB_g_L_24m", "MCV_fL_24m", "MCH_pg_24m"]
hd_fbc_features = [x[:-3] for x in hd_fbc_features_bl]
three_iron_features = ["HGB_g_L", "MCV_fL", "MCH_pg"]
difference_features_24m = getattr(feature_lists, "difference_features_24m")
ret_features = getattr(feature_lists, "ret_features")
pltf_features = getattr(feature_lists, "plt_f_features")
ip_qflags_features = getattr(feature_lists, "ip_qflags_features")
gaussian_features = getattr(feature_lists, "gaussian_features")

feature_lookup = {
    "non_blood": [],
    "HGB_bl": ["HGB_g_L_bl"],
    "standard_FBC_bl": standard_fbc_features_bl,
    "HD_FBC_bl": hd_fbc_features_bl,
    "three_iron_bl": three_iron_features_bl,
    "standard_FBC_24m": standard_fbc_features_24m,
    "HD_FBC_24m": hd_fbc_features_24m,
    "three_iron_24m": three_iron_features_24m,
    "standard_FBC_bl_24m": standard_fbc_features_bl + standard_fbc_features_24m,
    "HD_FBC_bl_24m": hd_fbc_features_bl + hd_fbc_features_24m,
    "HD_FBC_bl_24m_ferritin": hd_fbc_features_bl + hd_fbc_features_24m + ["FERR_bl"],
    "standard_FBC_bl_24m_ferritin": standard_fbc_features_bl
    + standard_fbc_features_24m
    + ["FERR_bl"],
    "standard_FBC_bl_24m_ferritin_trialarm": standard_fbc_features_bl
    + standard_fbc_features_24m
    + ["FERR_bl", "subject_trial_arm_severity"],
    "standard_FBC_bl_24m_delta_ferritin_trialarm": difference_features_24m
    + standard_fbc_features_24m
    + ["FERR_bl", "subject_trial_arm_severity"],
    "HD_FBC_bl_24m_ferritin_trialarm": hd_fbc_features_bl
    + hd_fbc_features_24m
    + ["FERR_bl", "subject_trial_arm_severity"],
    "standard_FBC_trialarm_deferral": standard_fbc_features_bl
    + ["subject_trial_arm_severity"],
    "HD_FBC_trialarm_deferral": hd_fbc_features_bl + ["subject_trial_arm_severity"],
    "HD_FBC_ferritin_trialarm_deferral": hd_fbc_features_bl
    + ["FERR_bl", "subject_trial_arm_severity"],
    "HD_FBC_sex": hd_fbc_features_bl,
    "three_iron_alltimepoints": three_iron_features,
    "standard_FBC_alltimepoints": hl_features,
    "HD_FBC_alltimepoints": hd_fbc_features,
    "HD_FBC_uncorr_alltimepoints": uncorr_features,
    "HD_FBC_sex_alltimepoints": hd_fbc_features,
    "HD_FBC_bl_ferritin_trialarm": hd_fbc_features_bl
    + ["FERR_bl", "subject_trial_arm_severity"],
    "HD_FBC_bl_24m_ferritin_trialarm": hd_fbc_features_bl
    + hd_fbc_features_24m
    + ["FERR_bl", "FERR_24m", "subject_trial_arm_severity"],
    "HD_FBC_noRET_noPLTF_noIP": list(
        set(uncorr_features)
        - set(ret_features)
        - set(pltf_features)
        - set(ip_qflags_features)
    ),
    "HD_FBC_noRET_noPLTF": list(
        set(uncorr_features) - set(ret_features) - set(pltf_features)
    ),
    "HD_FBC_noRET": list(set(uncorr_features) - set(ret_features)),
    "HD_FBC_noPLTF": list(set(uncorr_features) - set(pltf_features)),
    "HD_FBC_noIP": list(set(uncorr_features) - set(ip_qflags_features)),
    "HD_FBC_noRET_noIP": list(
        set(uncorr_features) - set(ret_features) - set(ip_qflags_features)
    ),
    "HD_FBC_noPLTF_noIP": list(
        set(uncorr_features) - set(pltf_features) - set(ip_qflags_features)
    ),
    "HD_FBC_noGaussian": list(set(uncorr_features) - set(gaussian_features)),
    "HD-FBC_noGaussian_noRET": list(
        set(uncorr_features) - set(gaussian_features) - set(ret_features)
    ),
    "HD-FBC_noGaussian_noRET_noPLTF": list(
        set(uncorr_features)
        - set(gaussian_features)
        - set(ret_features)
        - set(pltf_features)
    ),
    "HD-FBC_noGaussian_noRET_noPLTF_noIP": list(
        set(uncorr_features)
        - set(gaussian_features)
        - set(ret_features)
        - set(pltf_features)
        - set(ip_qflags_features)
    ),
    "HD-FBC_noGaussian_noIP": list(
        set(uncorr_features) - set(gaussian_features) - set(ip_qflags_features)
    ),
    "HD-FBC_noGaussian_noPLTF": list(
        set(uncorr_features) - set(gaussian_features) - set(pltf_features)
    ),
    "HD-FBC_noGaussian_noRET_noIP": list(
        set(uncorr_features)
        - set(gaussian_features)
        - set(ret_features)
        - set(ip_qflags_features)
    ),
    "HD-FBC_noGaussian_noPLTF_noIP": list(
        set(uncorr_features)
        - set(gaussian_features)
        - set(pltf_features)
        - set(ip_qflags_features)
    ),
}

multi_timepoint_featuresets = [
    "HD_FBC_bl_24m_ferritin",
    "HD_FBC_bl_24m_ferritin_trialarm",
    "standard_FBC_bl_24m_ferritin",
    "standard_FBC_bl_24m_ferritin_trialarm",
    "standard_FBC_bl_24m_delta_ferritin_trialarm",
    "standard_FBC_bl_24m_ferritin",
    "standard_FBC_alltimepoints",
    "three_iron_alltimepoints",
    "HD_FBC_alltimepoints",
    "HD_FBC_uncorr_alltimepoints",
    "HD_FBC_sex_alltimepoints",
    "HD_FBC_bl_ferritin_trialarm",
    "HD_FBC_bl_24m_ferritin_trialarm",
    "HD_FBC_noRET_noPLTF_noIP",
    "HD_FBC_noRET_noPLTF",
    "HD_FBC_noRET",
    "HD_FBC_noPLTF",
    "HD_FBC_noIP",
    "HD_FBC_noRET_noIP",
    "HD_FBC_noPLTF_noIP",
    "HD_FBC_noGaussian",
    "HD-FBC_noGaussian_noRET",
    "HD-FBC_noGaussian_noRET_noPLTF",
    "HD-FBC_noGaussian_noRET_noPLTF_noIP",
    "HD-FBC_noGaussian_noIP",
    "HD-FBC_noGaussian_noPLTF",
    "HD-FBC_noGaussian_noRET_noIP",
    "HD-FBC_noGaussian_noPLTF_noIP",
]

pgs_features = [
    "pgs_hct",
    "pgs_hgb",
    "pgs_mch",
    "pgs_mcv",
    "pgs_mpv",
    "pgs_pdw",
    "pgs_plt",
    "pgs_rdw",
    "pgs_ferritin",
    "hfe all reference",
    "hfe C282Y homo",
    "hfe H63D homo",
    "hfe S65C homo",
]

pretty_pgs_features = {
    "pgs_hct": "HCT PGS",
    "pgs_hgb": "HGB PGS",
    "pgs_mch": "MCH PGS",
    "pgs_mcv": "MCV PGS",
    "pgs_mpv": "MPV PGS",
    "pgs_pdw": "PDW PGS",
    "pgs_plt": "PLT PGS",
    "pgs_rdw": "RDW PGS",
    "pgs_ferritin": "Ferritin PGS",
    "hfe all reference": "HFE: all reference",
    "hfe C282Y homo": "HFE C282Y homozygous",
    "hfe H63D homo": "HFE H63D homozygous",
    "hfe S65C homo": "HFE S65C homozygous",
}
