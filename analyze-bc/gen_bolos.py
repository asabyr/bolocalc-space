import sys
import os
root_path="/Users/asabyr/Documents/software/bolocalc-space/"
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
import unpack as up
import subprocess
import numpy as np
import pandas as pd
import glob

class GenBolos:

    def __init__(self,bc_fp,exp_fp,band_edges,file_prefix, exp='specter_v1', tel='SPECTER', cam='BF', eta=0.7,force_sim = False):
        """ Accept a tuple of band edges, e.g. ([10,40], [40,80], ...) and write as tophats to the specified Bolocalc experiment directory with a default 0.7 detector efficiency"""

        self.bc_fp = bc_fp
        self.exp_fp = exp_fp
        self.file_prefix=file_prefix
        #self.freqs = np.arange(0.75*np.min(band_edges), 1.25*np.max(band_edges))
        self.freqs = np.arange(0.75*np.min(band_edges), 1.25*np.max(band_edges),0.1)
        self.cached_dict={}
        self.cached_dict_all={}
        if force_sim:
            self.band_edges = np.vstack(band_edges)
        else:
            self.band_edges = np.vstack(self.read_cache(band_edges))
        #print(self.band_edges)
        self.N_bands = len(self.band_edges)
        self.passbands = np.zeros( (self.N_bands, self.freqs.shape[0]))
        for b,edges in enumerate(self.band_edges):
            for j,freq in enumerate(self.freqs):
                if freq > edges[0] and freq < edges[1]:
                    self.passbands[b,j] = 1
        self.passbands *= eta
        self.band_centers = np.sum( self.freqs * self.passbands, axis=1) / np.sum(self.passbands, axis=1)

        self.pixel_sizes = self.pitch_estimate()

        #self.unpack = up.Unpack(self.file_prefix)
        #self.unpack.unpack_sensitivities(self.exp_fp)
        # for exp in self.unpack.sens_outputs.keys():
        #     if exp == 'Summary':
        #         continue
        #     self.exp = exp
        #     print(self.exp)
        #     for tel in self.unpack.sens_outputs[exp].keys():
        #         if tel == 'Summary':
        #             continue
        #         self.tel = tel
        #         print(self.tel)
        #         for cam in self.unpack.sens_outputs[exp][tel].keys():
        #             if cam == 'Summary':
        #                 continue
        #             self.cam = cam
        #             print(self.cam)

        # for exp in self.unpack.sens_outputs.keys():
        #     if exp == 'Summary':
        #         continue
        self.exp = exp
            # print(self.exp)
            # for tel in self.unpack.sens_outputs[exp].keys():
            #     if tel == 'Summary':
            #         continue
        self.tel = tel
                # print(self.tel)
                # for cam in self.unpack.sens_outputs[exp][tel].keys():
                #     if cam == 'Summary':
                #         continue
        self.cam = cam
                    # print(self.cam)

        self.cam_config = f'{self.exp_fp}{self.tel}/{self.cam}/config/'
        self.bands_fp = f'{self.cam_config}Bands/Detectors/'
        self.optics_fp = f'{self.cam_config}{self.file_prefix}optics.txt'
        self.channels_fp = f'{self.cam_config}{self.file_prefix}channels.txt'
        self.cmd_bolo = ['python3', self.bc_fp, self.exp_fp, f' --log_name {self.file_prefix}', f' --prefix {self.file_prefix}']

        self.optics_titles =     ("Element", "Temperature",
                "Absorption", "Reflection",
                "Thickness", "Index",
                "Loss Tangent", "Conductivity",
                "Surface Rough", "Spillover",
                "Spillover Temp", "Scatter Frac",
                "Scatter Temp")
        self.optics_band_cols = [2,3,9,11]
        self.optics_units = ('', '[K]', 'NA', 'NA', '[mm]', 'NA', '[e-4]', '[e6 S/m]', '[um RMS]', 'NA', '[K]', 'NA', '[K]')
        self.optics_stack = ('Primary', 'Secondary', 'Filter1', 'Filter2', 'Lyot', 'Filter3')
        self.det_titles =     ("Band ID", "Pixel ID",
                "Band Center", "Fractional BW",
                "Pixel Size", "Num Det per Wafer",
                "Num Waf per OT", "Num OT",
                "Waist Factor", "Det Eff",
                "Psat", "Psat Factor",
                "Carrier Index", "Tc",
                 "Tc Fraction", "Flink",
                 "Yield", "SQUID NEI",
                 "Bolo Resistance", "Read Noise Frac",
                 "G")
        self.det_units = ('', 'NA', '[GHz]', 'NA', '[mm]', 'NA', 'NA', 'NA', 'NA', 'NA', '[pW]', 'NA', 'NA', '[K]', 'NA', 'NA', 'NA', '[pA/rtHz]', '[Ohms]', 'NA', '[pW/K]')

    def read_cache(self, band_edges):
        # this method checks for bands that are already stored in the cache dictionary, saves those sensitivities, and returns the remaining bands to run
        try:
            cached_sens = np.load(f'{self.exp_fp}/{self.file_prefix}sens_out.npy', allow_pickle=True).item()
            #print(cached_sens)
            band_edges = [tuple(x) for x in band_edges]

            #print(band_edges)
            cnt_all = 0
            cnt= 0
            self.cached_dict_all = {}
            self.cached_dict={}
            saved_edges = []
            #print(cached_sens.keys())
            for band in cached_sens.keys():
                self.cached_dict_all[cnt_all] = {}
                self.cached_dict_all[cnt_all] = cached_sens[band]

                for j,nus in enumerate(band_edges):

                    if np.all(cached_sens[band]['Band Edges'] == nus):
                        saved_edges.append(nus)
                        self.cached_dict[cnt] = {}
                        self.cached_dict[cnt] = cached_sens[band]
                        cnt+=1

                cnt_all += 1

            rem_edges = list(set(band_edges) ^ set(saved_edges))

            def getKey(item):
                return item[0]
            rem_edges = np.array(sorted(rem_edges, key=getKey))

            if len(rem_edges) == 0:
                print('No new bands found, simulating all')
                self.cached_dict = {}
                self.cached_dict_all = {}
                return band_edges
            else:
                return rem_edges
        except:
            print('No cached dictionary found, simulating all input bands')
            return band_edges

    def pitch_estimate(self):
        ref_sizes = np.load(root_path+'/analyze-bc'+'/pixel_pitch.npy', allow_pickle=True).item()
        sizes = []
        for low_edge in self.band_edges[:,0]:
            freq_idx = find_nearest(ref_sizes['Freq'], low_edge)[0]
            sizes.append(ref_sizes['Size'][freq_idx])
        return sizes

    def write_bands(self):
        os.makedirs(self.bands_fp, exist_ok=True)
        #print(self.bands_fp+self.file_prefix+'*')
        if glob.glob(self.bands_fp+self.file_prefix+'*'):
            print(glob.glob(self.bands_fp+self.file_prefix+'*'))
            print("clearing bands")
            clear_bands = ["rm", self.bands_fp+self.file_prefix+"*"]
            run_cmd(' '.join(clear_bands))
            #print('cleared bands')
            #sys.exit()
        #print(self.passbands)
        for j,band in enumerate(self.passbands):
            #print("writing bands")
            #print(f'{self.bands_fp}{self.file_prefix}{self.cam}_{j+1}.txt')
            np.savetxt(f'{self.bands_fp}{self.file_prefix}{self.cam}_{j+1}.txt', np.c_[self.freqs,band])

    def write_optics_heading(self, entries, units = False):
        row = []
        for j,entry in enumerate(entries):
            if j in self.optics_band_cols:
                row.append(f'{entry:<{7*self.N_bands}s}')
            else:
                row.append(f'{entry:<{16}s}')
        if units:
            row[0] = '#' + row[0][1:]
        return ' | '.join(row)

    def write_optics_row(self, optic):
        param_row = []
        for j,param in enumerate(self.optics_titles):
            if param in ['Absorption','Reflection','Spillover','Scatter Frac']:
                if isinstance(self.calc_optic(optic,param,self.band_centers[0]), str):
                    param_row.append(f'{"NA":<{7*self.N_bands}s}')
                    continue
                band_vals = []
                for nu in self.band_centers:
                    val = self.calc_optic(optic,param,nu)
                    band_vals.append(f'{val:>0.3f}')
                band_vals = ', '.join(band_vals)
                param_row.append(f'[{band_vals}]')
            elif param in ['Temperature']:
                val = self.calc_optic(optic,param)
                param_row.append(f'{val:>0.3f}{"":<{11}s}')
            elif param in ['Element']:
                param_row.append(f'{optic:<{16}s}')
            else:
                param_row.append(f'{"NA":<{16}s}')
        return ' | '.join(param_row)

    def write_optics_table(self):
        init_row = '#*****Optical Chain*****'
        title_row = self.write_optics_heading(self.optics_titles)
        unit_row = self.write_optics_heading(self.optics_units, units=True)
        lines = [init_row, title_row, unit_row]
        for optic in self.optics_stack:
            lines.append(self.write_optics_row(optic))
        with open(self.optics_fp, 'w') as f:
            for line in lines:
                f.write(f'{line}\n#{"-"*len(title_row)}\n')

    def calc_optic(self, optic, param, nu=150):
        temps = {'Primary': 4.0, 'Secondary': 4.0, 'Filter1': 3.9, 'Filter2': 1.0, 'Lyot': 1.0, 'Filter3': 0.1}
        absorbs = {'Primary': nu**1.1 * 1e-5, 'Secondary': nu**1.1 * 1e-5, 'Filter1': 0.01, 'Filter2': 0.01, 'Lyot': 'NA', 'Filter3': 0.01}
        reflects = {'Primary': 0.0, 'Secondary': 0.0, 'Filter1': 0.05, 'Filter2': 0.05, 'Lyot': 0.0, 'Filter3': 0.05}
        spills = {'Primary': 0.0, 'Secondary': 0.0, 'Filter1': 0.0, 'Filter2': 0.0, 'Lyot': 0.0, 'Filter3': 0.0}
        scatter = {'Primary': 0.0, 'Secondary': 0.0, 'Filter1': 0.0, 'Filter2': 0.0, 'Lyot': 0.0, 'Filter3': 0.0}
        if param == 'Absorption':
            return absorbs[optic]
        elif param == 'Reflection':
            return reflects[optic]
        elif param == 'Temperature':
            return temps[optic]
        elif param == 'Spillover':
            return spills[optic]
        elif param == 'Scatter Frac':
            return scatter[optic]
        else:
            print('wrong parameter!')
            return

    def write_det_heading(self, entries, units=False):
        row = []
        for j,entry in enumerate(entries):
            row.append(f'{entry:<{17}s}')
        if units:
            row[0] = '#' + row[0][1:]
        return ' | '.join(row)

    def write_det_row(self, band_id, nu, pixel_size):
        det_dict = {"Band ID":band_id, "Pixel ID": 1,
                    "Band Center": 'BAND', "Fractional BW": 'NA',
                    "Pixel Size": pixel_size, "Num Det per Wafer": 1,
                    "Num Waf per OT": 1, "Num OT": 1,
                    "Waist Factor": 3.0, "Det Eff": 'NA',
                    "Psat": 'NA', "Psat Factor": 3.0,
                    "Carrier Index": 2.7, "Tc": 0.159,
                     "Tc Fraction": 'NA', "Flink": 1.0,
                     "Yield": 1, "SQUID NEI": 45.0,
                     "Bolo Resistance": 0.004, "Read Noise Frac": 'NA',
                     "G": 'NA'}
        param_row = []
        for key,val in det_dict.items():
            if isinstance(val,str):
                param_row.append(f'{val:<{17}s}')
            elif isinstance(val,float):
                param_row.append(f'{val:>0.3f}{"":<{12}s}')
            elif isinstance(val,int):
                param_row.append(f'{val:<{17}d}')
        return ' | '.join(param_row)

    def write_det_table(self):
        init_row = '#*****Detector Channels*****'
        title_row = self.write_det_heading(self.det_titles)
        unit_row = self.write_det_heading(self.det_units, units=True)
        lines = [init_row, title_row, unit_row]
        for j,nu in enumerate(self.band_centers):
            lines.append(self.write_det_row(j+1, nu, self.pixel_sizes[j]))
        with open(self.channels_fp, 'w') as f:
            for line in lines:
                f.write(f'{line}\n#{"-"*len(title_row)}\n')

    def write_experiment(self):
        self.write_optics_table()
        self.write_det_table()
        self.write_bands()

    def calc_bolos(self):
        self.write_experiment()
        #!{' '.join(self.cmd_bolo)}
        run_cmd(' '.join(self.cmd_bolo))

        self.unpack = up.Unpack(self.file_prefix)
        self.unpack.unpack_sensitivities(self.exp_fp)

        if len(self.cached_dict.keys())>0 and len(self.cached_dict_all.keys())>1:
            self.new_dict = self.cached_dict.copy()
            self.new_dict_all=self.cached_dict_all.copy()
            c = 1 + len(self.cached_dict.keys())
            c_all=1+len(self.cached_dict_all.keys())
            print('appending to existing sensitivity file')
        else:
            self.new_dict = {}
            self.new_dict_all={}
            c = 1
            c_all=1+len(self.cached_dict_all.keys())

        for j,band in enumerate(self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'].keys()):

            self.new_dict[j+c] = {}
            self.new_dict[j+c]['Center Frequency'] = self.band_centers[j]
            self.new_dict[j+c]['Band Edges'] = tuple(self.band_edges[j])
            self.new_dict[j+c]['Detector NET_CMB'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Detector NET_CMB'][0]
            self.new_dict[j+c]['Detector NET_RJ'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Detector NET_RJ'][0]
            self.new_dict[j+c]['Optical Power'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Optical Power'][0]

            self.new_dict_all[j+c_all] = {}
            self.new_dict_all[j+c_all]['Center Frequency'] = self.band_centers[j]
            self.new_dict_all[j+c_all]['Band Edges'] = tuple(self.band_edges[j])
            self.new_dict_all[j+c_all]['Detector NET_CMB'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Detector NET_CMB'][0]
            self.new_dict_all[j+c_all]['Detector NET_RJ'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Detector NET_RJ'][0]
            self.new_dict_all[j+c_all]['Optical Power'] = self.unpack.sens_outputs[self.exp][self.tel][self.cam]['All'][band]['Optical Power'][0]

        print(f'saving sensitivities to {self.exp_fp}{self.file_prefix}sens_out.npy')
        self.new_dict = self.sort_dict(self.new_dict)
        self.new_dict_all = self.sort_dict(self.new_dict_all)
        np.save(f'{self.exp_fp}{self.file_prefix}sens_out.npy', self.new_dict_all, allow_pickle=True)
        print(self.new_dict_all)
        return self.new_dict

    def sort_dict(self, sdict, key = 'Center Frequency'):
        # sort a dictionary by its 2nd key
        idxs, vals = [], []
        for j in sdict.keys():
            vals.append(sdict[j][key])
            idxs.append(j)
        idxs = np.array(idxs)
        vals = np.array(vals)
        sdict_sorted = {}
        for j,i in enumerate(np.argsort(vals)):
            sdict_sorted[j] = sdict[idxs[i]]
        return sdict_sorted

def run_cmd(cmd: str, stderr=subprocess.STDOUT) -> None:
    """Run a command in terminal

    Args:
        cmd (str): command to run in terminal
        stderr (subprocess, optional): Where the error has to go. Defaults to subprocess.STDOUT.

    Raises:
        e: Excetion of the CalledProcessError
    """
    out = None
    try:
        out = subprocess.check_output(
            [cmd],
            shell=True,
            stderr=stderr,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'ERROR {e.returncode}: {cmd}\n\t{e.output}',
              flush=True, file=sys.stderr)
        raise e
    print(out)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]
