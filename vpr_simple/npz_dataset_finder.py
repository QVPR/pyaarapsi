from os import listdir, stat
import numpy as np
import pandas as pd
from tqdm import tqdm


def gen_df(path, file_list=None, check_dataset=False):
    if file_list is None:
        file_list = listdir(path)
    dataset_dict = []

    for file in tqdm(file_list, disable = ~check_dataset):
        try:
            data  = np.load(path+file, allow_pickle=True)
        except:
            continue
        
        file_stats = stat(path+file)
        d = dict(file = file, file_size_MB = file_stats.st_size / (1024**2))

        if check_dataset:
            try:
                d.update(dict(
                    corrupted = False,
                    length = len(data['dataset'].item()['time']),
                    start_time = data['dataset'].item()['time'][0],
                    end_time = data['dataset'].item()['time'][-1],
                ))
            except EOFError:
                d.update(dict(
                    corrupted = True,
                    length = None,
                    start_time = 0,
                    end_time = 0,
                ))
        d.update(dict(
            **dict(data['params'].item())
        ))
        dataset_dict.append(d)
    
    df = pd.DataFrame(dataset_dict)
    return df


def file_finder(path, option_list_dict, file_list=None, return_df=False):
    # Option list dict should be in the format {'parameter name': [option1, option2, etc]}
    # example option_list_dict {'bag_name': ['s3_cw_1', 's3_cw_2'], 'img_dims': [[64, 64]]}
    df = gen_df(path, file_list)
    for key, value in option_list_dict.items():
        df = df[df[key].isin(value)]

    if return_df is False:
        return list(df['file'])
    return df

        
if __name__ == '__main__':
    path = input('path to query, defaults to `./data/compressed_sets/`: ') or './data/compressed_sets/'
    df = gen_df(path)
    print(df)

    print("Demonstrating finding a file from a specific bag, with specific image dimensions")
    file_list = file_finder(path,
                            {'bag_name': ['s4_ccw_1', 's4_ccw_2'], 'img_dims': [[64, 64]]})
    df = gen_df(path, file_list, check_dataset=True)
    print(df)
