import numpy as np


COLS_TO_REMOVE = [
    # Identifiers
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",

    # Administrative
    "QSTVER", "QSTLANG", "MSCODE", "CADULT", "LANDLINE", "NUMADULT", "HHADULT",
    "CSTATE", "FLSHTMY2", "HIVTSTD3",

    # Weighting variables
    "_WT2RAKE", "_STSTR", "_STRWT", "_RAWRAKE", "_DUALCOR", "_LLCPWT", "_CLLCPWT", "_DUALUSE",

    # Demographic variables not relevant for the prediction
    "_CRACE1", "_CPRACE", "_CHISPNC", "NUMHHOL2", "NUMPHON2", "CPDEMO1",

    # Variables only in the landline / phone survey
    "CTELENUM", "CTELNUM1", "CELLFON2", "CELLFON3", "PVTRESD1", "PVTRESD2", "STATERES",
    "COLGHOUS", "CCLGHOUS", "LADULT", "NUMWOMEN", "NUMMEN",

    # Highly correlated with other variables (>= 95%)
    "_DRDXAR1", "HTM4", "SMOKDAY2", "_PRACE1", "MAXVO2_", "_VEGETEX", "PADUR2_", "PADUR1_",
    "_FRUITEX", "_TOTINDA", "_DRNKWEK", "_RFSEAT3", "ASTHNOW", "_LMTSCL1", "_LMACT1",
    "_AIDTST3", "_LMTWRK1", "_PAINDX1", "_PA30021", "_EDUCAG", "_FRTLT1",
    "_AGE80", "_RACEGR3", "_AGE_G",

    # Missing values > 50%
    "BPMEDS", "DIABAGE2", "PREGNANT", "STOPSMK2", "LASTSMK2", "AVEDRNK2", "DRNK3GE5",
    "MAXDRNKS", "EXEROFT2", "EXERHMM2", "LMTJOIN3", "ARTHDIS2", "ARTHSOCL", "JOINPAIN",
    "IMFVPLAC", "WHRTST10", "PDIABTST", "PREDIAB1", "INSULIN", "BLDSUGAR", "FEETCHK2",
    "DOCTDIAB", "CHKHEMO3", "FEETCHK", "EYEEXAM", "DIABEYE", "DIABEDU", "CAREGIV1",
    "CRGVREL1", "CRGVLNG1", "CRGVHRS1", "CRGVPRB1", "CRGVPERS", "CRGVHOUS", "CRGVMST2",
    "CRGVEXPT", "VIDFCLT2", "VIREDIF3", "VIPRFVS2", "VINOCRE2", "VIEYEXM2", "VIINSUR2",
    "VICTRCT4", "VIGLUMA2", "VIMACDG2", "CIMEMLOS", "CDHOUSE", "CDASSIST", "CDHELP",
    "CDSOCIAL", "CDDISCUS", "WTCHSALT", "LONGWTCH", "DRADVISE", "ASTHMAGE", "ASATTACK",
    "ASERVIST", "ASDRVIST", "ASRCHKUP", "ASACTLIM", "ASYMPTOM", "ASNOSLEP", "ASTHMED3",
    "ASINHALR", "HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF", "RLIVPAIN", "RDUCHART",
    "RDUCSTRK", "ARTTODAY", "ARTHWGT", "ARTHEXER", "ARTHEDU", "TETANUS", "HPVADVC2",
    "HPVADSHT", "SHINGLE2", "HADMAM", "HOWLONG", "HADPAP2", "LASTPAP2", "HPVTEST",
    "HPLSTTST", "HADHYST2", "PROFEXAM", "LENGEXAM", "BLDSTOOL", "LSTBLDS3", "HADSIGM3",
    "HADSGCO1", "LASTSIG3", "PCPSAAD2", "PCPSADI1", "PCPSARE1", "PSATEST1", "PSATIME",
    "PCPSARS1", "PCPSADE1", "PCDMDECN", "SCNTMNY1", "SCNTMEL1", "SCNTPAID", "SCNTWRK1",
    "SCNTLPAD", "SCNTLWK1", "SXORIENT", "TRNSGNDR", "RCSGENDR", "RCSRLTN2", "CASTHDX2",
    "CASTHNO2", "EMTSUPRT", "LSATISFY", "ADPLEASR", "ADDOWN", "ADSLEEP", "ADENERGY",
    "ADEAT1", "ADFAIL", "ADTHINK", "ADMOVE", "MISTMNT", "ADANXEV", "PAFREQ2_",
    "_FLSHOT6", "_PNEUMO2",

    # Near constant features
    "_FRT16", "_VEG23",

    # Qualitative analysis
    "_RFHLTH", "POORHLTH", "_HCVU651", "BPHIGH4", "BLOODCHO", "CHOLCHK", "_RFCHOL",
    "ASTHMA3", "_LTASTH1", "_CASTHM1", "_LMTACT1",
    "WEIGHT2", "HEIGHT3", "HTIN4", "WTKG3", "_BMI5CAT", "_RFBMI5",
    "_MRACE1", "_HISPANC", "_RACEG21", "_RACE_G1", "_AGE65YR", "_INCOMG", "_CHLDCNT",
    "SMOKE100", "_RFSMOK3", "ALCDAY5", "DRNKANY5",
    "FRUITJU1", "FRUIT1", "FTJUDA1_", "FRUTDA1_", "_MISFRTN", "_FRTRESP",
    "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1", "BEANDAY_", "GRENDAY_", "ORNGDAY_",
    "VEGEDA1_", "_MISVEGN", "_VEGRESP", "_VEGLT1",
    "EXRACT11", "EXEROFT1", "EXERHMM1", "EXRACT21", "METVL11_", "METVL21_", "FC60_",
    "ACTIN11_", "ACTIN21_", "PAFREQ1_", "_MINAC11", "_MINAC21", "PAMISS1_", "PAMIN11_",
    "PAMIN21_", "PAVIG11_", "PAVIG21_", "_PACAT1", "_PA150R2", "_PA300R2", "_PAREC1",
    "_PASTAE1", "STRENGTH", "_PASTRNG", "_RFSEAT2",
]


# Codes that mean "Don't know / Refused / Missing" -> NaN
MISSING_CODES = {
    # 7 = Don't know, 9 = Refused
    "GENHLTH": [7, 9], "HLTHPLN1": [7, 9], "PERSDOC2": [7, 9], "MEDCOST": [7, 9],
    "CHECKUP1": [7, 9], "TOLDHI2": [7, 9], "CVDSTRK3": [7, 9], "CHCSCNCR": [7, 9],
    "CHCOCNCR": [7, 9], "CHCCOPD1": [7, 9], "HAVARTH3": [7, 9], "ADDEPEV2": [7, 9],
    "CHCKIDNY": [7, 9], "DIABETE3": [7, 9], "RENTHOM1": [7, 9], "VETERAN3": [7, 9],
    "INTERNET": [7, 9], "QLACTLM2": [7, 9], "USEEQUIP": [7, 9], "BLIND": [7, 9],
    "DECIDE": [7, 9], "DIFFWALK": [7, 9], "DIFFDRES": [7, 9], "DIFFALON": [7, 9],
    "USENOW3": [7, 9], "EXERANY2": [7, 9], "FLUSHOT6": [7, 9], "PNEUVAC3": [7, 9],
    "HIVTST6": [7, 9],
    "SEATBELT": [7, 8, 9],  # 8 = Never drive or ride in a car

    # 9 = Refused / Don't know / Missing
    "MARITAL": [9], "EDUCA": [9], "EMPLOY1": [9],
    "_RFHYPE5": [9], "_CHOLCHK": [9], "_ASTHMS1": [9], "_RACE": [9],
    "_SMOKER3": [9], "_RFBING5": [9], "_RFDRHV5": [9],

    # 77 = Don't know, 99 = Refused
    "PHYSHLTH": [77, 99], "MENTHLTH": [77, 99], "INCOME2": [77, 99],
    "CHILDREN": [99],

    # Other codes
    "_AGEG5YR": [14],     # Don't know / Refused / Missing
    "DROCDY3_": [900],    # Don't know / Refused / Missing
    "STRFREQ_": [99900],  # Don't know / Refused / Missing
}

# Codes that mean "None" -> 0
ZERO_CODES = {
    "CHILDREN": [88], "MENTHLTH": [88], "PHYSHLTH": [88],
}

NUMERICAL_COLS = [
    "PHYSHLTH", "MENTHLTH", "CHILDREN", "_BMI5", "DROCDY3_",
    "_FRUTSUM", "_VEGESUM", "STRFREQ_", "PA1MIN_", "PA1VIGM_",
]

ORDINAL_COLS = [
    "GENHLTH", "CHECKUP1", "EDUCA", "INCOME2", "_AGEG5YR",
    "USENOW3", "SEATBELT", "_SMOKER3", "_CHOLCHK", "_ASTHMS1",
]

NOMINAL_COLS = [
    # Binary (yes/no)
    "HLTHPLN1", "MEDCOST", "TOLDHI2", "CVDSTRK3", "CHCSCNCR", "CHCOCNCR",
    "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY", "SEX", "VETERAN3",
    "INTERNET", "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK",
    "DIFFDRES", "DIFFALON", "EXERANY2", "FLUSHOT6", "PNEUVAC3", "HIVTST6",
    "_RFHYPE5", "_RFBING5", "_RFDRHV5",

    # Multi-class
    "PERSDOC2", "DIABETE3", "MARITAL", "RENTHOM1", "EMPLOY1", "_RACE",
]


def feature_selection(x, labels):
    """
    Remove columns, handle special codes and build the variable-type masks.

    Args:
        x: array of shape (N, D) with the original features (x_train or x_test).
        labels: list of the D original feature names (headers[1:]).

    Returns:
        x_clean: array of shape (N, D') with the selected and cleaned features.
        labels_clean: list of the D' remaining feature names.
        masks: dict of boolean arrays of length D', with keys
               "numerical", "ordinal" and "nominal".
    """
    x = np.array(x, dtype=float, copy=True)  # never modify the original array
    labels = list(labels)

    if x.shape[1] != len(labels):
        raise ValueError(
            f"x has {x.shape[1]} columns but labels has {len(labels)} names"
        )

    keep = [i for i, name in enumerate(labels) if name not in COLS_TO_REMOVE]
    x = x[:, keep]
    labels = [labels[i] for i in keep]
    col_idx = {name: i for i, name in enumerate(labels)}

    for name, codes in MISSING_CODES.items():
        if name in col_idx:
            column = x[:, col_idx[name]]
            column[np.isin(column, codes)] = np.nan

    for name, codes in ZERO_CODES.items():
        if name in col_idx:
            column = x[:, col_idx[name]]
            column[np.isin(column, codes)] = 0

    labels_array = np.array(labels)
    numerical_mask = np.isin(labels_array, NUMERICAL_COLS)
    ordinal_mask = np.isin(labels_array, ORDINAL_COLS)
    nominal_mask = np.isin(labels_array, NOMINAL_COLS)

    masks = {
        "numerical": numerical_mask,
        "ordinal": ordinal_mask,
        "nominal": nominal_mask,
    }

    return x, labels, masks