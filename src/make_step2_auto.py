import numpy as np
from scipy.ndimage.filters import maximum_filter
import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
from utils import Rod, R2atom
import subprocess

MONOMER_LIST = ["BTBT","naphthalene","anthracene","tetracene","pentacene","hexacene","demo"]
############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A1,A2,A3,phi=0.0,isFF=False):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv('~/Working/step2_auto/{}/assets/monomer.csv'.format(monomer_name))
    atoms_array_xyzR=df_mono[['X','Y','Z','R']].values
    
    ex = np.array([1.,0.,0.]); ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])

    xyz_array = atoms_array_xyzR[:,:3]
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)
    xyz_array = np.matmul(xyz_array,Rod(ey,A1).T)
    xyz_array = xyz_array + T_vec
    R_array = atoms_array_xyzR[:,3].reshape((-1,1))
    
    if monomer_name in MONOMER_LIST:
        return np.concatenate([xyz_array,R_array],axis=1)
    
    elif monomer_name=='mono-C9-BTBT':
        #alkylの基準
        C0_index = 16 #BTBT骨格の端
        C1_index = 23 #アルキルの根本
        
        C0=xyz_array[C0_index]
        C1=xyz_array[C1_index]
        
        #phi1に関するalkylの軸
        n1=C1-C0
        n1/=np.linalg.norm(n1)
        
        #alkyl回転・分子1作成
        xyz_array[C1_index:] = np.matmul((xyz_array[C1_index:]-C0),Rod(n1,phi).T) + C0
        
        if isFF:
            FFconfig_array=df_mono[['q','sig','eps']].values
            return np.concatenate([xyz_array,R_array,FFconfig_array],axis=1)
        else:
            return np.concatenate([xyz_array,R_array],axis=1)
    
    else:
        raise RuntimeError('invalid monomer_name={}'.format(monomer_name))
        

def get_xyzR_lines(xyzR_array,file_description):
    lines = [     
        '%mem=15GB\n',
        '%nproc=40\n',
        '#P TEST pbepbe/6-311G** EmpiricalDispersion=GD3 counterpoise=2\n',
        '\n',
        file_description+'\n',
        '\n',
        '0 1 0 1 0 1\n'
    ]
    mol_len = len(xyzR_array)//2
    atom_index = 0
    mol_index = 0
    for x,y,z,R in xyzR_array:
        atom = R2atom(R)
        mol_index = atom_index//mol_len + 1
        line = '{}(Fragment={}) {} {} {}\n'.format(atom,mol_index,x,y,z)     
        lines.append(line)
        atom_index += 1
    return lines        

def make_gjf_xyz(auto_dir,monomer_name,params_dict):##計算する際のジョブファイル作成
    a_ = params_dict['a']; b_ = params_dict['b']; #c = np.array([params_dict['cx'],params_dict['cy'],params_dict['cz']])
    A1 = 0; A2 = 0; A3 = params_dict['theta']
    ##get_monomer_xyzRで分子の座標データの作成
    gij_xyz_lines1 = ['$ RunGauss\n']
    gij_xyz_lines2 = ['$ RunGauss\n']
    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A1,A2,A3)##対称性よりs-pの面内は片方で十分
    z_list=[np.round(z,1) for z in np.linspace(np.round(0,1),np.round(4,1),int(np.round(np.round(4,1)/0.1))+1)]
    for z in z_list:
        if a_>b_:
            monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,z,A1,A2,A3)
        else:
            monomer_array_p1 = get_monomer_xyzR(monomer_name,a_,0,z,A1,A2,A3)
    
        monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,z,-A1,A2,-A3)##誘導体はtが等価でないから4つつくる
   
        dimer_array_t1 = np.concatenate([monomer_array_i,monomer_array_t1])##2分子の座標データを結合
        dimer_array_p1 = np.concatenate([monomer_array_i,monomer_array_p1])
    
        file_description = '{}_A1={}_A2={}_A3={}'.format(monomer_name,int(A1),int(A2),round(A3,2))##ファイル名の角度部分　位置情報はそれぞれ後で加える
        line_list_dimer_p1 = get_xyzR_lines(dimer_array_p1,file_description+'_p1')##2分子の計算ファイルの文章部分の作成　位置情報をファイル名に加えた
        line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1')
        
##上で作った文章部分を結合して1か所の計算をまとめる
        
        gij_xyz_lines1 = gij_xyz_lines1 + line_list_dimer_t1 + ['\n\n--Link1--\n']##p*2+t*4でいける
        gij_xyz_lines2 = gij_xyz_lines2 + line_list_dimer_p1 + ['\n\n--Link1--\n']
    
    gij_xyz_lines1 = gij_xyz_lines1 + ['\n\n\n']
    gij_xyz_lines2 = gij_xyz_lines2 + ['\n\n\n']
    file_base_name = ''
    file_base_name += monomer_name
    file_base_name += '_step2_'
    file_base_name += 'a={}_b={}_theta={}_'.format(a_,b_,A3)
    file_name1 = file_base_name
    file_name2 = file_base_name
    file_name1 +='1.inp'
    file_name2 +='2.inp'
    os.makedirs(os.path.join(auto_dir,'gaussian'),exist_ok=True)##auto_dir\gaussianに格納　そこへのパスを指定している
    gij_xyz_path1 = os.path.join(auto_dir,'gaussian',file_name1)##ファイルへのパス
    gij_xyz_path2 = os.path.join(auto_dir,'gaussian',file_name2)##ファイルへのパス
    with open(gij_xyz_path1,'w') as f1: ##.inpファイルを作成してそこに上で作った計算をまとめたものを書き込む
        f1.writelines(gij_xyz_lines1)##.inpファイルの作成完了
    with open(gij_xyz_path2,'w') as f2: ##.inpファイルを作成してそこに上で作った計算をまとめたものを書き込む
        f2.writelines(gij_xyz_lines2)##.inpファイルの作成完了
    return file_base_name

def get_one_exe(file_basename,machine_type):
    
    #mkdir
    if machine_type==1:
        gr_num = 1; mp_num = 40
    elif machine_type==2:
        gr_num = 2; mp_num = 52
    cc_list=[
        '#!/bin/sh \n',
         '#$ -S /bin/sh \n',
         '#$ -cwd \n',
         '#$ -V \n',
         '#$ -q gr{}.q \n'.format(gr_num),
         '#$ -pe OpenMP {} \n'.format(mp_num),
         '\n',
         'hostname \n',
         '\n',
         'export g16root=/home/g03 \n',
         'source $g16root/g16/bsd/g16.profile \n',
         '\n',
         'export GAUSS_SCRDIR=/home/scr/$JOB_ID \n',
         'mkdir /home/scr/$JOB_ID \n',
         '\n',
         'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
         '\n',
         'rm -rf /home/scr/$JOB_ID \n',
         '\n',
         '\n',
         '#sleep 5 \n'
#          '#sleep 500 \n'
            ]

    return cc_list

def exec_gjf(auto_dir, monomer_name, params_dict,isTest):
    inp_dir = os.path.join(auto_dir,'gaussian')
    print(params_dict)

    file_base_name = make_gjf_xyz(auto_dir, monomer_name, params_dict)

    file_basename1 = file_base_name
    file_basename2 = file_base_name
    file_basename1 +='1'
    file_basename2 +='2'

    cc_list1 = get_one_exe(file_basename1,machine_type=1)
    sh_filename1 = file_basename1 + '.r1'
    sh_path1 = os.path.join(inp_dir,sh_filename1)
    with open(sh_path1,'w') as f:
        f.writelines(cc_list1)
    if not(isTest):
        subprocess.run(['qsub',sh_path1])
    log_file_name1 = file_basename1 + '.log'

    cc_list2 = get_one_exe(file_basename2,machine_type=2)
    sh_filename2 = file_basename2 + '.r1'
    sh_path2 = os.path.join(inp_dir,sh_filename2)
    with open(sh_path2,'w') as f:
        f.writelines(cc_list2)
    if not(isTest):
        subprocess.run(['qsub',sh_path2])
    log_file_name2 = file_basename2 + '.log'
    return log_file_name1,log_file_name2
    
############################################################################################