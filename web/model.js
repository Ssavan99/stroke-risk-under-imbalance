window.STROKE_MODEL = {
  "schema": 1,
  "model": "logistic_regression (unweighted)",
  "preprocess": {
    "numeric_features": [
      "age",
      "avg_glucose_level",
      "bmi",
      "hypertension",
      "heart_disease"
    ],
    "numeric_fill": {
      "age": 45.0,
      "avg_glucose_level": 91.945,
      "bmi": 28.0,
      "hypertension": 0.0,
      "heart_disease": 0.0
    },
    "indicator_features": [
      "bmi"
    ],
    "scaler_mean": [
      43.3532876712,
      106.317167319,
      28.8838307241,
      0.0971135029354,
      0.054060665362,
      0.0415851272016
    ],
    "scaler_scale": [
      22.5940515948,
      45.2541164425,
      7.76296877807,
      0.296112259934,
      0.226137369364,
      0.199639185525
    ],
    "categorical_features": [
      "gender",
      "ever_married",
      "work_type",
      "Residence_type",
      "smoking_status"
    ],
    "categorical_fill": {
      "gender": "Female",
      "ever_married": "Yes",
      "work_type": "Private",
      "Residence_type": "Urban",
      "smoking_status": "never smoked"
    },
    "categories": {
      "gender": [
        "Female",
        "Male",
        "Other"
      ],
      "ever_married": [
        "No",
        "Yes"
      ],
      "work_type": [
        "Govt_job",
        "Never_worked",
        "Private",
        "Self-employed",
        "children"
      ],
      "Residence_type": [
        "Rural",
        "Urban"
      ],
      "smoking_status": [
        "Unknown",
        "formerly smoked",
        "never smoked",
        "smokes"
      ]
    }
  },
  "coefficients": [
    1.62990942507,
    0.151247615744,
    0.0658869809085,
    0.117287653801,
    0.0343298226998,
    0.284948876569,
    -0.00151525310255,
    0.0327301558952,
    -0.0189278502391,
    0.138291246984,
    -0.126004194431,
    0.00146166971383,
    -0.0665334880977,
    0.0683053912232,
    -0.213111213144,
    0.222164692858,
    -0.0244967894398,
    0.0367838419933,
    -0.145580188632,
    0.104035571503,
    -0.0805798245445,
    0.134411494227
  ],
  "intercept": -4.04249607941,
  "feature_names": [
    "num__age",
    "num__avg_glucose_level",
    "num__bmi",
    "num__hypertension",
    "num__heart_disease",
    "num__missingindicator_bmi",
    "cat__gender_Female",
    "cat__gender_Male",
    "cat__gender_Other",
    "cat__ever_married_No",
    "cat__ever_married_Yes",
    "cat__work_type_Govt_job",
    "cat__work_type_Never_worked",
    "cat__work_type_Private",
    "cat__work_type_Self-employed",
    "cat__work_type_children",
    "cat__Residence_type_Rural",
    "cat__Residence_type_Urban",
    "cat__smoking_status_Unknown",
    "cat__smoking_status_formerly smoked",
    "cat__smoking_status_never smoked",
    "cat__smoking_status_smokes"
  ],
  "operating_point": {
    "threshold": 0.0473807822312,
    "sensitivity": 0.8,
    "specificity": 0.744855967078,
    "precision": 0.138888888889,
    "npv": 0.986376021798,
    "false_positive_rate": 0.255144032922,
    "number_needed_to_screen": 7.2,
    "tp": 40,
    "fp": 248,
    "fn": 10,
    "tn": 724
  },
  "sweep": [
    {
      "threshold": 0.000676283158257,
      "sensitivity": 1.0,
      "specificity": 0.0,
      "precision": 0.0489236790607,
      "accuracy": 0.0489236790607,
      "flagged": 1022,
      "flagged_fraction": 1.0
    },
    {
      "threshold": 0.00985012840433,
      "sensitivity": 0.92,
      "specificity": 0.42695473251,
      "precision": 0.0762852404643,
      "accuracy": 0.451076320939,
      "flagged": 603,
      "flagged_fraction": 0.590019569472
    },
    {
      "threshold": 0.0190239736504,
      "sensitivity": 0.88,
      "specificity": 0.565843621399,
      "precision": 0.0944206008584,
      "accuracy": 0.581213307241,
      "flagged": 466,
      "flagged_fraction": 0.455968688845
    },
    {
      "threshold": 0.0281978188965,
      "sensitivity": 0.86,
      "specificity": 0.634773662551,
      "precision": 0.108040201005,
      "accuracy": 0.645792563601,
      "flagged": 398,
      "flagged_fraction": 0.389432485323
    },
    {
      "threshold": 0.0373716641426,
      "sensitivity": 0.86,
      "specificity": 0.697530864198,
      "precision": 0.127596439169,
      "accuracy": 0.705479452055,
      "flagged": 337,
      "flagged_fraction": 0.329745596869
    },
    {
      "threshold": 0.0465455093886,
      "sensitivity": 0.8,
      "specificity": 0.742798353909,
      "precision": 0.137931034483,
      "accuracy": 0.745596868885,
      "flagged": 290,
      "flagged_fraction": 0.283757338552
    },
    {
      "threshold": 0.0557193546347,
      "sensitivity": 0.8,
      "specificity": 0.775720164609,
      "precision": 0.15503875969,
      "accuracy": 0.776908023483,
      "flagged": 258,
      "flagged_fraction": 0.252446183953
    },
    {
      "threshold": 0.0648931998808,
      "sensitivity": 0.78,
      "specificity": 0.803497942387,
      "precision": 0.169565217391,
      "accuracy": 0.802348336595,
      "flagged": 230,
      "flagged_fraction": 0.225048923679
    },
    {
      "threshold": 0.0740670451269,
      "sensitivity": 0.74,
      "specificity": 0.824074074074,
      "precision": 0.177884615385,
      "accuracy": 0.819960861057,
      "flagged": 208,
      "flagged_fraction": 0.203522504892
    },
    {
      "threshold": 0.083240890373,
      "sensitivity": 0.72,
      "specificity": 0.842592592593,
      "precision": 0.190476190476,
      "accuracy": 0.836594911937,
      "flagged": 189,
      "flagged_fraction": 0.184931506849
    },
    {
      "threshold": 0.092414735619,
      "sensitivity": 0.7,
      "specificity": 0.851851851852,
      "precision": 0.195530726257,
      "accuracy": 0.844422700587,
      "flagged": 179,
      "flagged_fraction": 0.175146771037
    },
    {
      "threshold": 0.101588580865,
      "sensitivity": 0.7,
      "specificity": 0.87037037037,
      "precision": 0.217391304348,
      "accuracy": 0.862035225049,
      "flagged": 161,
      "flagged_fraction": 0.157534246575
    },
    {
      "threshold": 0.110762426111,
      "sensitivity": 0.66,
      "specificity": 0.880658436214,
      "precision": 0.221476510067,
      "accuracy": 0.869863013699,
      "flagged": 149,
      "flagged_fraction": 0.145792563601
    },
    {
      "threshold": 0.119936271357,
      "sensitivity": 0.6,
      "specificity": 0.890946502058,
      "precision": 0.220588235294,
      "accuracy": 0.876712328767,
      "flagged": 136,
      "flagged_fraction": 0.133072407045
    },
    {
      "threshold": 0.129110116603,
      "sensitivity": 0.58,
      "specificity": 0.900205761317,
      "precision": 0.230158730159,
      "accuracy": 0.884540117417,
      "flagged": 126,
      "flagged_fraction": 0.123287671233
    },
    {
      "threshold": 0.138283961849,
      "sensitivity": 0.58,
      "specificity": 0.91049382716,
      "precision": 0.25,
      "accuracy": 0.894324853229,
      "flagged": 116,
      "flagged_fraction": 0.113502935421
    },
    {
      "threshold": 0.147457807096,
      "sensitivity": 0.44,
      "specificity": 0.925925925926,
      "precision": 0.234042553191,
      "accuracy": 0.902152641879,
      "flagged": 94,
      "flagged_fraction": 0.0919765166341
    },
    {
      "threshold": 0.156631652342,
      "sensitivity": 0.42,
      "specificity": 0.932098765432,
      "precision": 0.241379310345,
      "accuracy": 0.907045009785,
      "flagged": 87,
      "flagged_fraction": 0.0851272015656
    },
    {
      "threshold": 0.165805497588,
      "sensitivity": 0.38,
      "specificity": 0.938271604938,
      "precision": 0.240506329114,
      "accuracy": 0.91095890411,
      "flagged": 79,
      "flagged_fraction": 0.0772994129159
    },
    {
      "threshold": 0.174979342834,
      "sensitivity": 0.34,
      "specificity": 0.945473251029,
      "precision": 0.242857142857,
      "accuracy": 0.915851272016,
      "flagged": 70,
      "flagged_fraction": 0.0684931506849
    },
    {
      "threshold": 0.18415318808,
      "sensitivity": 0.3,
      "specificity": 0.953703703704,
      "precision": 0.25,
      "accuracy": 0.921722113503,
      "flagged": 60,
      "flagged_fraction": 0.0587084148728
    },
    {
      "threshold": 0.193327033326,
      "sensitivity": 0.24,
      "specificity": 0.956790123457,
      "precision": 0.222222222222,
      "accuracy": 0.921722113503,
      "flagged": 54,
      "flagged_fraction": 0.0528375733855
    },
    {
      "threshold": 0.202500878572,
      "sensitivity": 0.24,
      "specificity": 0.961934156379,
      "precision": 0.244897959184,
      "accuracy": 0.926614481409,
      "flagged": 49,
      "flagged_fraction": 0.0479452054795
    },
    {
      "threshold": 0.211674723818,
      "sensitivity": 0.2,
      "specificity": 0.965020576132,
      "precision": 0.227272727273,
      "accuracy": 0.92759295499,
      "flagged": 44,
      "flagged_fraction": 0.0430528375734
    },
    {
      "threshold": 0.220848569064,
      "sensitivity": 0.18,
      "specificity": 0.972222222222,
      "precision": 0.25,
      "accuracy": 0.933463796477,
      "flagged": 36,
      "flagged_fraction": 0.0352250489237
    },
    {
      "threshold": 0.23002241431,
      "sensitivity": 0.16,
      "specificity": 0.974279835391,
      "precision": 0.242424242424,
      "accuracy": 0.934442270059,
      "flagged": 33,
      "flagged_fraction": 0.03228962818
    },
    {
      "threshold": 0.239196259556,
      "sensitivity": 0.16,
      "specificity": 0.977366255144,
      "precision": 0.266666666667,
      "accuracy": 0.937377690802,
      "flagged": 30,
      "flagged_fraction": 0.0293542074364
    },
    {
      "threshold": 0.248370104802,
      "sensitivity": 0.16,
      "specificity": 0.981481481481,
      "precision": 0.307692307692,
      "accuracy": 0.941291585127,
      "flagged": 26,
      "flagged_fraction": 0.0254403131115
    },
    {
      "threshold": 0.257543950048,
      "sensitivity": 0.14,
      "specificity": 0.98353909465,
      "precision": 0.304347826087,
      "accuracy": 0.942270058708,
      "flagged": 23,
      "flagged_fraction": 0.0225048923679
    },
    {
      "threshold": 0.266717795295,
      "sensitivity": 0.12,
      "specificity": 0.984567901235,
      "precision": 0.285714285714,
      "accuracy": 0.942270058708,
      "flagged": 21,
      "flagged_fraction": 0.0205479452055
    },
    {
      "threshold": 0.275891640541,
      "sensitivity": 0.12,
      "specificity": 0.986625514403,
      "precision": 0.315789473684,
      "accuracy": 0.944227005871,
      "flagged": 19,
      "flagged_fraction": 0.0185909980431
    },
    {
      "threshold": 0.285065485787,
      "sensitivity": 0.12,
      "specificity": 0.989711934156,
      "precision": 0.375,
      "accuracy": 0.947162426614,
      "flagged": 16,
      "flagged_fraction": 0.0156555772994
    },
    {
      "threshold": 0.294239331033,
      "sensitivity": 0.1,
      "specificity": 0.990740740741,
      "precision": 0.357142857143,
      "accuracy": 0.947162426614,
      "flagged": 14,
      "flagged_fraction": 0.013698630137
    },
    {
      "threshold": 0.303413176279,
      "sensitivity": 0.1,
      "specificity": 0.992798353909,
      "precision": 0.416666666667,
      "accuracy": 0.949119373777,
      "flagged": 12,
      "flagged_fraction": 0.0117416829746
    },
    {
      "threshold": 0.312587021525,
      "sensitivity": 0.1,
      "specificity": 0.992798353909,
      "precision": 0.416666666667,
      "accuracy": 0.949119373777,
      "flagged": 12,
      "flagged_fraction": 0.0117416829746
    },
    {
      "threshold": 0.321760866771,
      "sensitivity": 0.08,
      "specificity": 0.993827160494,
      "precision": 0.4,
      "accuracy": 0.949119373777,
      "flagged": 10,
      "flagged_fraction": 0.00978473581213
    },
    {
      "threshold": 0.330934712017,
      "sensitivity": 0.08,
      "specificity": 0.993827160494,
      "precision": 0.4,
      "accuracy": 0.949119373777,
      "flagged": 10,
      "flagged_fraction": 0.00978473581213
    },
    {
      "threshold": 0.340108557263,
      "sensitivity": 0.06,
      "specificity": 0.993827160494,
      "precision": 0.333333333333,
      "accuracy": 0.948140900196,
      "flagged": 9,
      "flagged_fraction": 0.00880626223092
    },
    {
      "threshold": 0.349282402509,
      "sensitivity": 0.06,
      "specificity": 0.995884773663,
      "precision": 0.428571428571,
      "accuracy": 0.950097847358,
      "flagged": 7,
      "flagged_fraction": 0.00684931506849
    },
    {
      "threshold": 0.358456247755,
      "sensitivity": 0.06,
      "specificity": 0.995884773663,
      "precision": 0.428571428571,
      "accuracy": 0.950097847358,
      "flagged": 7,
      "flagged_fraction": 0.00684931506849
    },
    {
      "threshold": 0.367630093001,
      "sensitivity": 0.06,
      "specificity": 0.995884773663,
      "precision": 0.428571428571,
      "accuracy": 0.950097847358,
      "flagged": 7,
      "flagged_fraction": 0.00684931506849
    },
    {
      "threshold": 0.376803938247,
      "sensitivity": 0.06,
      "specificity": 0.996913580247,
      "precision": 0.5,
      "accuracy": 0.951076320939,
      "flagged": 6,
      "flagged_fraction": 0.00587084148728
    },
    {
      "threshold": 0.385977783494,
      "sensitivity": 0.06,
      "specificity": 0.997942386831,
      "precision": 0.6,
      "accuracy": 0.952054794521,
      "flagged": 5,
      "flagged_fraction": 0.00489236790607
    },
    {
      "threshold": 0.39515162874,
      "sensitivity": 0.04,
      "specificity": 0.997942386831,
      "precision": 0.5,
      "accuracy": 0.951076320939,
      "flagged": 4,
      "flagged_fraction": 0.00391389432485
    },
    {
      "threshold": 0.404325473986,
      "sensitivity": 0.04,
      "specificity": 0.998971193416,
      "precision": 0.666666666667,
      "accuracy": 0.952054794521,
      "flagged": 3,
      "flagged_fraction": 0.00293542074364
    },
    {
      "threshold": 0.413499319232,
      "sensitivity": 0.04,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.953033268102,
      "flagged": 2,
      "flagged_fraction": 0.00195694716243
    },
    {
      "threshold": 0.422673164478,
      "sensitivity": 0.04,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.953033268102,
      "flagged": 2,
      "flagged_fraction": 0.00195694716243
    },
    {
      "threshold": 0.431847009724,
      "sensitivity": 0.04,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.953033268102,
      "flagged": 2,
      "flagged_fraction": 0.00195694716243
    },
    {
      "threshold": 0.44102085497,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.450194700216,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.459368545462,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.468542390708,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.477716235954,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.4868900812,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.496063926446,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.505237771693,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.514411616939,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.523585462185,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.532759307431,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    },
    {
      "threshold": 0.541933152677,
      "sensitivity": 0.02,
      "specificity": 1.0,
      "precision": 1.0,
      "accuracy": 0.952054794521,
      "flagged": 1,
      "flagged_fraction": 0.000978473581213
    }
  ],
  "performance": {
    "pr_auc": 0.257486349247,
    "pr_auc_no_skill": 0.0489236790607,
    "roc_auc": 0.842510288066,
    "brier": 0.0410717530415,
    "prevalence": 0.0489236790607,
    "n_test": 1022,
    "n_test_positive": 50,
    "mean_predicted_risk": 0.0468337669038
  },
  "dataset": {
    "rows": 5110,
    "positives": 249,
    "prevalence": 0.0487279843444
  }
};
