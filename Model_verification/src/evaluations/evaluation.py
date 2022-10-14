import os
import matplotlib.pyplot as plt
import time
import json
import numpy as np
import pandas as pd


class evaluation:
    def __init__(self, base_folder=None):
        if base_folder:
            self.result_folder = os.path.join(os.getcwd(), "reports", base_folder)
        else:
            self.result_folder = os.path.join(os.getcwd(), "reports")

    def make_run_folder(self, ctx, exp_run=None):
        #        datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
        #        folder_name = datetime + "_" + ctx
        #        self.run_folder = os.path.join(self.result_folder, folder_name)
        #        os.makedirs(self.run_folder, exist_ok = True)
        try:
            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
            if exp_run:
                folder_name = datetime + "_" + ctx + "_run_" + str(exp_run)
            else:
                folder_name = datetime + "_" + ctx
            self.run_folder = os.path.join(self.result_folder, folder_name)
            os.makedirs(self.run_folder)
        except:
            rand_int = np.random.randint(1, 10)
            time.sleep(1)
            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
            folder_name = datetime + "_" + ctx
            self.run_folder = os.path.join(self.result_folder, folder_name)
            os.makedirs(self.run_folder)

    def get_run_folder(self):
        return self.run_folder

    def save_figure(self, figure, name: str, subfolder=None):
        name = name + ".png"
        if subfolder:
            os.makedirs(os.path.join(self.run_folder, subfolder), exist_ok=True)
            figure.savefig(os.path.join(self.run_folder, subfolder, name))
        else:
            figure.savefig(os.path.join(self.run_folder, name))

    def save_json(self, res_dict: dict, name: str, subfolder=None):
        name = name + ".json"
        if subfolder:
            os.makedirs(os.path.join(self.run_folder, subfolder), exist_ok=True)
            with open(os.path.join(self.run_folder, subfolder, name), "w") as out_file:
                json.dump(res_dict, out_file, indent=4)
        else:
            with open(os.path.join(self.run_folder, name), "w") as out_file:
                json.dump(res_dict, out_file, indent=4)

    def save_csv(self, data, name: str, subfolder=None):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        name = name + ".csv"
        if subfolder:
            os.makedirs(os.path.join(self.run_folder, subfolder), exist_ok=True)
            data.to_csv(os.path.join(self.run_folder, subfolder, name))
        else:
            data.to_csv(os.path.join(self.run_folder, name))
