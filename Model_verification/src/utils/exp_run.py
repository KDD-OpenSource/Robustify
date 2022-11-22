import os
import time


class exp_run:
    def __init__(self, base_folder=None):
        if base_folder:
            self.result_folder = os.path.join(os.getcwd(), "reports", base_folder)
        else:
            self.result_folder = os.path.join(os.getcwd(), "reports")

    def make_run_folder(self, ctx, run_number=None):
        try:
            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
            if run_number:
                folder_name = datetime + "_" + ctx + "_run_" + str(run_number)
            else:
                folder_name = datetime + "_" + ctx
            self.run_folder = os.path.join(self.result_folder, folder_name)
            os.makedirs(self.run_folder)
        except:
            time.sleep(1)
            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
            folder_name = datetime + "_" + ctx
            self.run_folder = os.path.join(self.result_folder, folder_name)
            os.makedirs(self.run_folder)

    def get_run_folder(self):
        return self.run_folder
