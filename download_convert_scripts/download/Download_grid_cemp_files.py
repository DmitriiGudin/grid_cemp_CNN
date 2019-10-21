# Created by Dmitrii Gudin (U Notre Dame).

import numpy as np
import os
import time
import utils
import paramiko

def Run(dir_in, dir_out, fname_list):
    start_time = time.time()
    def Print(s): utils.Print(s, start_time)
    def Print(s): print time.time()-start_time, "s: ", s
    for f, N in zip(fname_list, range(0,len(fname_list))):
        category = f[:5]
        if not os.path.exists(dir_in + "/" + category):
            os.makedirs(dir_in + "/" + category)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('nuit.phys.nd.edu', username='dgudin', password='Emlaskermlasc4')
        sftp = ssh.open_sftp()
        sftp.get(dir_out+"/"+f, dir_in+"/"+category+"/"+f)
        sftp.close()
        ssh.close()
        Print("File " + f + " copied.   -----   " + str(N) +" / " + str(len(fname_list)) + ",   " + str(int(float(N/len(fname_list)))) + ".")    
 
if __name__ == '__main__':
    dir_in = "/afs/crc.nd.edu/user/d/dgudin/starnet/data/grid_cemp_R2000_clean"
    dir_out = "/emc3/grid_cemp/R2000_clean"
    fname_list = np.loadtxt("grid_cemp_file_list.txt", delimiter=None, dtype=str)
    Run(dir_in, dir_out, fname_list)
