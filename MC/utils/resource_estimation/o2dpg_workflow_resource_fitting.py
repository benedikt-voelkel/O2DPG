#!/usr/bin/env python3

import sys
import argparse
from os.path import exists, join
from os import makedirs
from copy import deepcopy
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


RESOURCES = ["pss", "uss", "cpu"]
#FEATURES = ["ns", "eCM", "j"]
FEATURES = ["ns", "eCM", "interactionRate", "j"]


DEFAULT_PARAMETERS = {"interactionRate": 10000,
                      "eCM": 900,
                      "ns": 110,
                      "j": 16}

def transform_feature_values(feature_dict):
    """
    for the interaction rate, we do it statically for now, in the future we might need a dynamic approach
    """
    if "interactionRate" in feature_dict:
        feature_dict["interactionRate"] = 1 / np.array(feature_dict["interactionRate"])


def get_res_per_task(df):
    """
    Construct a dictionary of the form
    {"task_name1": {"cpu": [forFeatureValue1, forFeatureValue2, ...],
                    "pss": [forFeatureValue1, forFeatureValue2, ...],
                    "uss": [forFeatureValue1, forFeatureValue2, ...]},
     "task_name2": {"cpu": [forFeatureValue1, forFeatureValue2, ...],
                    "pss": [forFeatureValue1, forFeatureValue2, ...],
                    "uss": [forFeatureValue1, forFeatureValue2, ...]},
     "task_name3": ...
    }

    Return that dictionary along with a list of the feature values
    """

    # skim down the dataframe already as much as we can

    # get the unique task names
    df = df[RESOURCES + FEATURES + ["name", "id"]]
    task_names = df["name"].unique()
    df_ids = df["id"].unique()

    df_dict = {key: [] for key in RESOURCES + FEATURES + ["name"]}

    for df_id in df_ids:
        df_filt_id = df.query(f"id == {df_id}")
        # get the feature values for this one
        features = {}
        for key, value in df_filt_id.iloc[0].items():
            if key in FEATURES:
                features[key] = value
        for task in task_names:
            df_filt = df_filt_id.query(f"name == '{task}'")
            if not len(df_filt):
                continue
            resources = {key: max(df_filt[key].values) for key in RESOURCES}
            for key, value in features.items():
                df_dict[key].append(value)
            for res, value in resources.items():
                df_dict[res].append(value)
            df_dict["name"].append(task)

    return pd.DataFrame(df_dict)


def fit_model(model, X, y):
    model.fit(X, y)

    params = np.append(model.intercept_, model.coef_)
    # predictions = model.predict(X)

    # # newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    # # MSE = (sum((y - predictions)**2)) / (len(newX) - len(newX.columns))

    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    # # Note if you don't want to use a DataFrame replace the two lines above with
    # # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    # var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    # sd_b = np.sqrt(var_b)
    # ts_b = params/ sd_b

    # p = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

    return 0

class ResourceFitter:
    """
    Class wrapping the fitting and prediction of resources
    """
    def __init__(self):
        self.fit_dict = None

    def fit(self, df):
        # we want to fit for each single task
        tasks = df["name"].unique()
        # collect parameters in a dictionary
        # start with feature names; the seperate parameters will be added per task and resource
        fit_dict = {"feature_names": FEATURES,
                    "feature_name_to_id": {key: i for i, key in enumerate(FEATURES)},
                    "default_values": DEFAULT_PARAMETERS}
        # this is the per-task dictionary
        per_task_dict = {task: {} for task in tasks}
        # add to fit_dict so it will be serialised
        fit_dict["tasks"] = per_task_dict
        if "interactionRate" in FEATURES:
            df["interactionRate"] = 1 / df["interactionRate"]

        for task in tasks:
            if task != "tpcdigi":
                continue
            df_task = df.query(f"name == '{task}'")
            print(df_task)
            feature_values = df_task[FEATURES].values

            tree = DecisionTreeRegressor()
            tree.fit(feature_values, df_task[RESOURCES])

            print(tree.predict([[110, 900, 0.00001, 16]]))
            print(tree.predict([[300, 900, 0.00001, 16]]))
            print("---")
            #   p = fit_model(regr, feature_values, df_task[RESOURCES])
            regr = LinearRegression()
            p = fit_model(regr, feature_values, df_task[RESOURCES])
            print(regr.predict([[110, 13600, 0.00001, 16]]))
            print("---")
            nn = MLPRegressor(hidden_layer_sizes=(4,8), max_iter=1000, warm_start=True)
            scaler_nn = StandardScaler()
            scaler_nn.fit(feature_values)
            feature_values = scaler_nn.transform(feature_values)
            for _ in range(20):
                nn.fit(feature_values, df_task[RESOURCES])
            pred = scaler_nn.transform([[110, 900, 0.00001, 16]])
            print("---")
            print(nn.predict(pred))

            for res_name in RESOURCES[2:]:
                regr = LinearRegression()
                resources = df_task[res_name].values
                p = fit_model(regr, feature_values, resources)

                print(res_name, regr.predict([[110, 900, 0.00001, 16]]))
                # print(feature_values)
                # print(max(resources))
                # print()
                per_task_dict[task][res_name] = {"parameters": list(regr.coef_) + [regr.intercept_], "p": p}


        self.fit_dict = fit_dict

    def set_params(self, fit_dict):
        if isinstance(fit_dict, str):
            with open(fit_dict, "r") as f:
                self.fit_dict = json.load(f)
            return
        self.fit_dict = fit_dict

    def write(self, output):
        print(self.fit_dict)
        with open(output, "w") as f:
            json.dump(self.fit_dict, f, indent=2)

    def get_coef(self, task, metric):
        return np.array(self.fit_dict["tasks"][task][metric]["parameters"])

    def predict(self, task, feature_dict, metric, ignore_unknown=False):
        if not self.fit_dict:
            print("WARNING: No fit, cannot predict")
            return None

        # if there are multiple predictions done, we must not touch this dictionary because it is under the control of the user
        # ==> copy it
        feature_dict_ = deepcopy(feature_dict)
        feature_dict = {}
        known_features = self.fit_dict["feature_names"]
        default_values = self.fit_dict["default_values"]

        for feature_name in feature_dict_:
            # check if the user asks for feature that are known
            if feature_name not in known_features:
                if not ignore_unknown:
                    print(f"ERROR: Feature {feature_name} was not fitted earlier, known features are: {known_features}")
                    return None
                continue
            # make the dictionary with features we know
            feature_dict[feature_name] = feature_dict_[feature_name]

        for feature_name, default_value in default_values.items():
            # if something was not given, set the default
            feature_dict[feature_name] = feature_dict.get(feature_name, default_value)

        # now, potentially, some features need to be transformed
        #print(feature_dict)
        transform_feature_values(feature_dict)

        # sort the feature dict into a list
        feature_values_list = [np.array(feature_dict[feature_name]) for feature_name in known_features]

        coef = self.get_coef(task, metric)
        ret = np.sum([c * v for c, v in zip(coef[:len(feature_values_list)], feature_values_list)]) + coef[-1]

        return ret


def make_df(paths):
    # Load all dataframes from JSON and extract per-task resources
    dfs = [get_res_per_task(pd.read_json(df)) for df in paths]
    # concatenate all to one which will be used for the fit, strip down to what is needed
    #return pd.concat([pd.read_json(df) for df in paths])[FEATURES + RESOURCES + ["name"]]
    return pd.concat(dfs)[FEATURES + RESOURCES + ["name"]]


def fit(args):
    """
    Entrypoint
    """
    df = make_df(args.dfs)
    df = df.query("interactionRate > 200000")
    resource_fitter = ResourceFitter()
    resource_fitter.fit(df)
    resource_fitter.write(args.output)
    return 0


def make_dict_from_tokens(token_string):
    token_dict = {}
    key_value_pairs = token_string.split(",")

    for key_value_pair in key_value_pairs:
        try:
            key, value = key_value_pair.split("=")
            if key in token_dict:
                # in this case something was passed more than once
                print(f"ERROR: Parameter {key} already set to {token_dict[key]}")
                return None
            token_dict[key] = value
        except TypeError:
            print(f"ERROR: Invalid key-value pair {key_value_pair}")
            return None
    for key in list(token_dict.keys()):
        if ".." in token_dict[key]:
            low, up = token_dict[key].split("..")
            low, up = (float(low), float(up))
            token_dict[key] = np.arange(low, up, (up - low) / 1000)
            continue
        try:
            token_dict[key] = float(token_dict[key])
        except (TypeError, ValueError):
            continue
    return token_dict


def predict(args):
    """
    Entrypoint
    """
    if (not args.dfs and not args.json_params) or (args.dfs and args.json_params):
        print("ERROR: Need EITHER a list of DataFrames OR the parameter JSON from the fit process")
        return 1
    token_dict = make_dict_from_tokens(args.features)
    resource_fitter = ResourceFitter()
    if args.dfs:
        df = make_df(args.dfs)
        resource_fitter.fit(df)
    else:
        resource_fitter.set_params(args.json_params)

    print(resource_fitter.predict(args.task, token_dict, args.metric))

    return 0


def plot(args):
    """
    This plots data and fit for one parameter of interest
    """
    resource_fitter = ResourceFitter()
    resource_fitter.set_params(args.json_params)

    token_dict = make_dict_from_tokens(args.features)
    feature_of_interest = None
    for feat, value in token_dict.items():
        try:
            if len(value) > 1:
                feature_of_interest = feat
                break
        except TypeError:
            pass
    df = get_res_per_task(pd.read_json(args.df))

    # find constant features
    # get the feature values that are constant
    output = join(args.output, feature_of_interest)
    if not exists(output):
        makedirs(output)

    for task in df["name"].unique():
        df_skim = df.query(f"name == '{task}'")
        for res in RESOURCES:
            x = df_skim[feat].values
            if len(x) < 2:
                continue
            y = df_skim[res].values
            output_this = join(output, f"fit_{task}_{res}.png")
            fig, ax = plt.subplots(figsize=(20, 20))
            x_fit = token_dict[feature_of_interest].copy()
            ax.plot(x_fit, resource_fitter.predict(task, token_dict, res), label=f"fit", lw=3)
            ax.plot(x, y, lw=0, ms=30, marker="o", label="data")
            ax.set_xlabel(feat, fontsize=40)
            ax.set_ylabel(res, fontsize=40)
            ax.tick_params("both", labelsize=40)
            ax.legend(loc="best", fontsize=30)
            fig.suptitle(f"{task}, {res} vs. {feat}", fontsize=50)
            fig.tight_layout()
            fig.savefig(output_this)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Derive some scaling functions from O2DPG resource comparisons")
    sub_parsers = parser.add_subparsers(dest="command")

    fit_parser = sub_parsers.add_parser("fit", help="Fit extracted pipeline_metric files")
    fit_parser.set_defaults(func=fit)
    fit_parser.add_argument("--dfs", nargs="*", help="paths to JSONised dataframes", required=True)
    fit_parser.add_argument("-o", "--output", help="filename for parameter JSON with fits", default="o2dpg_sim_resource_parameters.json")

    predict_parser = sub_parsers.add_parser("predict", help="predict resources for a given task")
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("-t", "--task", help="task to derive resources for", required=True)
    predict_parser.add_argument("-m", "--metric", help="resource to be predicted", required=True)
    predict_parser.add_argument("--features", help="\",\"-separated 2-tuples, e.g. j=8,interactionRate=50000,eCM=4000", required=True)
     # one of the following 2 must be given
    predict_parser.add_argument("--dfs", nargs="*", help="paths to JSONised dataframes")
    predict_parser.add_argument("--json-params", dest="json_params", help="paths to parameter JSON")

    plot_parser = sub_parsers.add_parser("plot", help="plot a dataframe where ONE feature has been varied")
    plot_parser.set_defaults(func=plot)
    plot_parser.add_argument("--json-params", dest="json_params", help="JSON with fit parameters", required=True)
    plot_parser.add_argument("--df", help="path to JSONised dataframe", required=True)
    plot_parser.add_argument("--features", help="\",\"-separated 2-tuples, e.g. j=8,interactionRate=50000,eCM=900..14000, where one (in this case eCM) is given as a range to cover the data", required=True)
    plot_parser.add_argument("-o", "--output", help="plot everything to this directory", default="o2dpg_plot_sim_scaling")

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
