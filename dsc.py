from carbontracker import tracker, parser
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from helper_methods import *
import os
import argparse

# QuantileTransformer returns initial column names and not output names after passing through ColumnTransformer
# This is expected behaviour, and the warnings clutter the output
warnings.filterwarnings(category=FutureWarning, action="ignore")
# Same for SimpleImputer
warnings.filterwarnings(category=UserWarning, action="ignore")

plt.style.use('fivethirtyeight')
pd.set_option("display.max.columns", 20)
pd.set_option("display.precision", 2)


def read_data(f="Data/DSC_2021_Training.xlsx", sample=None):
    X_train, X_test, y_train, y_test = split_train_test(f, sample=sample)  # also writes Train.csv and Test.csv
    train_df = pd.read_csv('Data/Train.csv').drop(['LoanID', 'Unnamed: 0'], axis=1)  # for column statistics

    # test_df = pd.read_csv('Data/Test.csv').drop(['LoanID', 'Unnamed: 0'], axis=1)
    return X_train, X_test, y_train, y_test, train_df


def process(X_train, X_test, y_train, train_df,
            max_card=20, oversample=.3, max_skew=7., seed=987514):
    SKEWED_COLS = get_skewed_cols(X_train, max_skew)
    NOMINAL = ['Type', 'Managing_Sales_Office_Nbr', 'Postal_Code_L',
               'Product_Desc', 'CREDIT_TYPE_CD', 'FINANCIAL_PRODUCT_TYPE_CD',
               'INDUSTRY_CD_3', 'INDUSTRY_CD_4',
               'ACCOUNT_PURPOSE_CD', 'A2_MARITAL_STATUS_CD', 'A2_EMPLOYMENT_STATUS_CD',
               'A2_RESIDENT_STATUS_CD']  # features that will get normal nominal treatment
    CONTINUOUS = ['BEH_SCORE_AVG_A1', 'BEH_SCORE_AVG_A2', 'BEH_SCORE_MIN_A2', 'FHS_SCORE_AVG', 'FHS_SCORE_LATEST',
                  'CASH_FLOW_AMT', 'FREE_CASH_FLOW_AMT', 'A1_AVG_POS_SALDO_PROF_1_AMT',
                  'Invest_Amt', 'Original_loan_Amt', 'MTHS_FIRST_PCX_COREPRIV_CNT', 'A1_TOT_DEB_INTEREST_PROF_6_AMT',
                  'A2_MTHS_SNC_LAST_LIQ_PRIV_CNT', 'A2_MTHS_SNC_FIRST_COREPROF_CNT', 'A1_TOT_DEB_INTEREST_PROF_1_AMT',
                  'MTHS_IN_BUSINESS_CNT', 'A2_MTHS_SNC_LAST_LIQ_SAVE_CNT', 'A1_NEGAT_TRANS_COREPROF_CNT',
                  'A1_OVERDRAWN_DAYS_PROF_24_CNT', 'A1_OVERDRAWN_DAYS_PROF_6_CNT', 'A1_AVG_NEG_SALDO_PROF_3_AMT',
                  'A1_AVG_POS_SALDO_PROF_12_AMT', 'CASHFLOW_MONTHLY_CREDIT_RT',
                  'MONTHLY_CREDIT_AMT_TOT', 'A2_ANNUAL_INCOME_AMT_scale',
                  'A2_AVG_NEG_SALDO_PRIV_12_AMT_scale', 'A2_AVG_POS_SALDO_SAVINGS_12_AMT_scale',
                  'A2_TOTAL_EMPLOYMENT_MONTHS_CNT_scale'
                  # ,MTHS_SNC_LAST_REFUSAL_CNT
                  ]  # features that will get normal continuous treatment
    LOW_CARD = [col for col in NOMINAL if len(train_df[col].unique()) <= max_card]
    HIGH_CARD = [col for col in NOMINAL if len(train_df[col].unique()) > max_card]
    FILL_W_ZERO = ['A1_TOT_STND_PAYMNT_INT_PROF_CNT',  # if no info, assume no standing payments
                   'MTHS_SNC_1ST_REC_CNT'  # assume they're not in database yet
                   ]
    ONEHOT_CONT = ['MONTHS_SINCE_LAST_REFUSAL_CNT']

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse=True, handle_unknown="ignore"))
    ])

    onehot_cont_pipeline = Pipeline([
        # One-Hot encodes a continuous feature if it has a value or not (e.g. amount of months since last refusal)
        ("OnehotCont", OneHotEncoderContinuous())
    ])

    high_card_pipeline = Pipeline([
        ("high_card", NamedWOEEncoder())
    ])

    skewed_pipeline = Pipeline([
        # Applies a log to columns with a skew over MAX_SKEW
        ("skewed_columns", RectifySkewedColumns()),
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    numerical_pipeline = Pipeline([
        # Fills in mean and tries to scale to a normal distribution with QuantileTransformer()
        ("imputer", SimpleImputer(strategy="mean")),
        # rf performs worse with median, lr&knn too, but better confusion matrix
        # ("imputer", KNNImputer()),  # no substantial difference with SimpleImputer()
        ("scaler", QuantileTransformer(output_distribution='normal'))
    ])

    imbalance_pipeline = Pipeline([
        # Tackles imbalance for target class
        ("oversampler", ADASYN(sampling_strategy=oversample)),  # solid improvement
        ("undersampler", RandomUnderSampler(sampling_strategy=.5, random_state=seed)  # minor improvement
         )
    ])  # up- and undersampling (only for training set)

    preprocessing_transformers = ColumnTransformer([
        ('rectify_skewed_cols', skewed_pipeline, SKEWED_COLS),
        ('OneHotCont', onehot_cont_pipeline, ONEHOT_CONT),
        ("filling_low_card", low_card_pipeline, LOW_CARD),
        ("high_card_preprocessor", high_card_pipeline, HIGH_CARD),
        ("numerical_preprocessor", numerical_pipeline, CONTINUOUS),
        ("nan_means_zero", Pipeline([("imput_zero", SimpleImputer(strategy='constant', fill_value=0)),
                                     ("scaler", QuantileTransformer(output_distribution='normal'))]), FILL_W_ZERO)
    ])  # full preprocessing pipeline

    X_train = preprocessing_transformers.fit_transform(X_train, y_train)
    X_s, y_s = imbalance_pipeline.fit_resample(X_train, y_train)
    X_test_t = preprocessing_transformers.transform(X_test)

    pd.DataFrame(data=X_s, columns=preprocessing_transformers.get_feature_names_out()) \
        .to_csv('Data/Train_processed.csv')
    pd.DataFrame(data=X_test_t, columns=preprocessing_transformers.get_feature_names_out()) \
        .to_csv("Data/Test_processed.csv")

    return X_s, y_s, X_test_t, preprocessing_transformers


def dummy_train(X_s, y_s, X_test, y_test, preprocessing_transformers):
    model = LogisticRegression().fit(X_s, y_s)
    score = model.score(X_test, y_test)
    plot_confusion_matrix(model, "dummy_logistic_regression", X_test, y_test, score)
    plot_roc(model, "dummy_logistic_regression", X_test, y_test)
    show_importance(model, "dummy logistic regression", colnames=get_feature_names(preprocessing_transformers))
    make_prediction(model, preprocessing_transformers, "dummy_logistic_regression")


def train_and_tune(X_s, y_s, X_test, y_test, preprocessing_transformers, n_folds, seed):
    MODELS = {
        # 'lr_dummy': LogisticRegression,
        'lr': LogisticRegression,
        'knn': KNeighborsClassifier,
        'rf': RandomForestClassifier
    }
    HYPERPARAM1 = {
        "rf": {"n_estimators": [100, 1000, 1500, 2000],
               "max_depth": [10, 50, 100],
               "min_samples_split": [2, 10],
               "random_state": [seed]
               },
        "lr": {"penalty": ["l2"],
               "C": [10 ** e for e in np.arange(-3, 3, dtype=float)],
               # needs to be float to avoid ValueError bug in numpy
               "max_iter": [10, 100, 500, 1000]
               },
        "knn": {"n_neighbors": [10, 20, 50, 100, 500, 1000],
                "p": [1, 2]
                },
    }
    HYPERPARAM2 = {
        "rf": {"n_estimators": [2000, 2500, 3000],  # 2000 < 3000
               "max_depth": [50, 70, 90],  # 60 < 70 < 100
               "min_samples_split": [2],
               "random_state": [seed]
               },
        "lr": {"penalty": ["l2"],
               "C": [10 ** e for e in np.linspace(-1, 1, 5)],
               "max_iter": [80, 100, 120]
               },
        "knn": {"n_neighbors": [10],
                "p": [1]
                },
    }

    for hyperparam in [HYPERPARAM1, HYPERPARAM2]:
        for model_name in MODELS.keys():
            # param_grid = {model+'__'+key: val for key, val in HYPERPARAM[model].items()}  # adapt param grid names
            param_grid = hyperparam[model_name]
            gs = GridSearchCV(MODELS[model_name](), param_grid, cv=KFold(n_folds, random_state=seed, shuffle=True),
                              n_jobs=7, verbose=3, scoring="roc_auc")
            gs.fit(X_s, y_s)
            score = gs.score(X_test, y_test)
            pd.DataFrame(gs.cv_results_).to_csv("Model_Info/{}.csv".format(model_name))
            plot_confusion_matrix(gs, model_name, X_test, y_test, score)
            m = gs.best_estimator_
            plot_roc(m, model_name, X_test, y_test)
            show_importance(m, model_name, colnames=get_feature_names(preprocessing_transformers))
            make_prediction(m, preprocessing_transformers, model_name)


def run_w_carbontracker(func, suffix):
    """
    Decorator function to run some method while logging the power output with CarbonTracker
    """

    def inner(*args, **kwargs):
        if not os.path.exists('./AutomationOutputs'):
            os.mkdir('./AutomationOutputs')
        t = tracker.CarbonTracker(epochs=1, log_dir="./AutomationOutputs/{}/".format(suffix),
                                  update_interval=.5)
        t.epoch_start()

        func(*args, **kwargs)

        t.epoch_end()
        if not os.path.exists('./AutomationOutputs/{}'.format(suffix)):
            os.mkdir('./AutomationOutputs/{}'.format(suffix))
        parser.print_aggregate(log_dir="./AutomationOutputs/{}/".format(suffix))

        t.stop()
        return func

    return inner


def run(max_card=20, n_folds=5, oversample=.3, max_skew=7., seed=987514,
        n_explore=1, n_process=1, n_train=1,
        sample=0):
    assert all([n >= 1 for n in (n_explore, n_process, n_train)]), \
        "n_explore, n_process and n_train must all be larger than 1"
    if sample != 0:
        X_train, X_test, y_train, y_test, train_df = read_data("Data/DSC_2021_Training.xlsx", sample=sample)
    else:
        X_train, X_test, y_train, y_test, train_df = read_data("Data/DSC_2021_Training.xlsx")

    ##############################################################################################
    plot_distribution_of_features(train_df)
    X_s, y_s, X_test, preprocessing_transformers = process(X_train, X_test, y_train, train_df,
                                                           max_card, oversample, max_skew, seed)
    dummy_train(X_s, y_s, X_test, y_test, preprocessing_transformers)
    plot_distribution_of_features(pd.DataFrame(X_s, columns=preprocessing_transformers.get_feature_names_out()))
    plt.close('all')
    # iteratively improving data processing
    for _ in range(n_process - 1):
        X_s, y_s, X_test, preprocessing_transformers = process(X_train, X_test, y_train, train_df,
                                                               max_card, oversample, max_skew, seed)
        dummy_train(X_s, y_s, X_test, y_test, preprocessing_transformers)  # checking effect on AUC
        plot_distribution_of_features(train_df)
        plt.close('all')
    # training
    for _ in range(n_train):
        train_and_tune(X_s, y_s, X_test, y_test, preprocessing_transformers, n_folds, seed)
    return 0


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description='run a simulated data science project')
    argument_parser.add_argument('--suffix', default=1, help='index of the runfile, suffix of logs')
    argument_parser.add_argument('--sample', default=0, help="subsample the dataset", type=int)
    argument_parser.add_argument('--use_ct', default=False, const=True, nargs='?',
                                 help="Run the automated code while logging the power usage "
                                      "with CarbonTracker")
    args = argument_parser.parse_args()

    if args.use_ct:
        print("Measuring power using carbontracker and running script")
        run_w_carbontracker(run, suffix=args.suffix)(sample=args.sample)
    else:
        print("Measuring power using RAPL interface and running script")
        run(sample=args.sample)
