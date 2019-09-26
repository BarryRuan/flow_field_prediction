import os
import numpy as np

PATH = os.path.realpath(os.path.dirname(__file__))
PASI_DATA_DIR = os.path.join(PATH, 'Hs')
NEGA_DATA_DIR = os.path.join(PATH, 'Ls')
VELOCITY_TRAIN_DIR = os.path.join(PATH, 'data/velocity')
DIRECT_TRAIN_DIR = os.path.join(PATH, 'data/direction')
MAG_TRAIN_DIR = os.path.join(PATH, 'data/magnitude')


def generate_mag_vx_direction(dat, filenum, if_pasi):
    # 1. Transform the dat into its corresponding magnitude map,
    #    v_x map and direction map. 
    # 2. Write new maps into new files
    f = open(dat, 'r')
    lines = f.readlines()
    f.close()
    mt = np.zeros(25*25).reshape(25,25)
    direct = np.zeros(25*25*2).reshape(25,25,2)
    velocity = np.zeros(25*25*2).reshape(25,25,2)

    # Get corresponding data
    for i in range(625):
        if if_pasi:
            v = np.array([float(a) for a in lines[i+3].split(' ')[2:]])
        else:
            v = np.array([float(a) for a in lines[i].split('\t')[2:]])
        mag = np.linalg.norm(v)
        mt[i//25, i%25] = mag
        if (not v[0] == 0.0) or (not v[1] == 0.0):
            direct[i//25, i%25] = v/mag 
        velocity[i//25, i%25] = v

    if filenum == 1:
        file_name = '1'
    else:
        file_name = str(filenum)
    print("writing %s"%(file_name))

    # Magnitude map
    f = open(os.path.join(MAG_TRAIN_DIR, file_name), 'w')
    for line in mt: 
        f.writelines(' '.join([str(round(a,4)) for a in line]) + '\n')
    f.close()

    # Direction map
    ff = open(os.path.join(DIRECT_TRAIN_DIR, file_name), 'w')
    for line in direct: 
        dt_list = []
        for dt in line:
            dt_list.append(','.join([str(round(a,4)) for a in dt]))
        ff.writelines(' '.join(dt_list) + '\n')
    ff.close()

    # Velocity map
    fff = open(os.path.join(VELOCITY_TRAIN_DIR, file_name), 'w')
    for line in velocity: 
        dt_list = []
        for dt in line:
            dt_list.append(','.join([str(round(a,4)) for a in dt]))
        fff.writelines(' '.join(dt_list) + '\n')
    fff.close()

def process_data(data_dir, num_start=1, if_pasi=True):
    # Get all files
    data_dirs = os.listdir(data_dir)
    data_dirs = [a for a in data_dirs if a[0] == 'B']
    def dir_key(dirname):
        return [int(dirname[1:dirname.find('_')])]
    def dir_key1(dirname):
        return [int(dirname[1:dirname.find('.')])]
    data_dirs.sort(key=dir_key)
    i = num_start 
    j = 0
    for dir in data_dirs:
        if j == 100:
            break
        dir_name = os.path.join(data_dir, dir)
        dat_files = os.listdir(dir_name)
        dat_files= [a for a in dat_files if a[0] == 'B']
        print("data in %s will be mapped to training data from %d to %d" % (dir, (i), (i)+100))
        dat_files.sort(key=dir_key1)
        for dat in dat_files:
            if not dat[0] == 'B': 
                continue
            generate_mag_vx_direction(os.path.join(dir_name, dat), i, if_pasi)
            i += 1
        j += 1
    return i

if __name__ == '__main__':
    print("Processing pasitive data...")
    i = process_data(PASI_DATA_DIR, True)
    print("Processing negative data...")
    process_data(NEGA_DATA_DIR, i,  False)
