# Created by Dmitrii Gudin (U Notre Dame).

import numpy as np
import os
import time
import paramiko

def Run(dir_in, dir_out, fname_list):
    start_time = time.time()
    def Print(s): print time.time()-start_time, "s: ", s
    for f, N in zip(fname_list, range(0,len(fname_list))):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('nuit.phys.nd.edu', username='dgudin', password='Emlaskermlasc4')
        sftp = ssh.open_sftp()
        sftp.get(dir_out+"/"+f, dir_in+"/"+f)
        sftp.close()
        ssh.close()
        Print("File " + f + " copied.   -----   " + str(N) +" / " + str(len(fname_list)) + ",   " + str(int(float(N/len(fname_list)))) + ".")    
 
if __name__ == '__main__':
    dir_in = "/scratch365/dgudin/SDSS_spectra"
    dir_out = "/emc3/sspp/fits/MP"
    fname_list = np.loadtxt("/afs/crc.nd.edu/user/d/dgudin/starnet/SSPP_spectra_file_list.txt", delimiter=None, dtype=str)
    Run(dir_in, dir_out, fname_list)
