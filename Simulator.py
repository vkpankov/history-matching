import typing
import numpy as np
import subprocess
from ecl.summary import EclSum
import shutil

class Simulator:
        
    def init_dirs(self, errfunc):
        for i in range(0, self.threads_count):
            shutil.rmtree(f"{self.model_path}_{i}", ignore_errors=True) 
            shutil.copytree(self.model_path, f"{self.model_path}_{i}")

    def __init__(self, model_path: str, main_file_name: str, perm_file_name: str, porosity_file_name: str, summary_file_name: str, threads_count: int, init_dirs: bool = False):
        self.model_path: str = model_path
        self.main_file_name: str = main_file_name
        self.perm_file_name: str = perm_file_name
        self.porosity_file_name: str = porosity_file_name
        self.summary_file_name: str = summary_file_name
        self.threads_count: int = threads_count
        if(init_dirs):
            self.init_dirs(None)

    def __generate_file(self, values: np.ndarray, file_name: str, header: str):
        np.savetxt(file_name, values, header=header, fmt='%f', footer='/', comments='')

    def run_simulator(self, perm_porosity_maps):
        cmd = []
        for ind, (permmap, porositymap) in enumerate(perm_porosity_maps):
            self.__generate_file(permmap, f'{self.model_path}_{ind}/{self.perm_file_name}', 'PERMX')
            self.__generate_file(porositymap, f'{self.model_path}_{ind}/{self.porosity_file_name}', 'PORO')           
            cmd.append(['flow',  f'{self.model_path}_{ind}/{self.main_file_name}'])
        procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL) for c in cmd]
        for proc in procs:
            proc.wait()
        summaries = []
        for x in range(0, len(perm_porosity_maps)):
            summary = EclSum(f"{self.model_path}_{x}/{self.summary_file_name}", lazy_load=False)
            summaries.append(summary)
        return summaries

    def run_simulator_source_dir(self, perm_map, porosity_map):
        self.__generate_file(perm_map, f'{self.model_path}/{self.perm_file_name}', "PERMX")
        self.__generate_file(porosity_map, f'{self.model_path}/{self.porosity_file_name}', "PORO")
        cmd=['flow',  f'{self.model_path}/{self.main_file_name}']
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
        proc.wait()
        summary = EclSum(f"{self.model_path}/{self.summary_file_name}", include_restart=False, lazy_load=False)
        return summary