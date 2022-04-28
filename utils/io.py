import pandas as pd
import numpy as np
import torch
import boto3

print(torch.__version__)


n_cpus = 1
use_cuda = torch.cuda.is_available()


def get_files_list(bucket_name, path):
    """
    get all files names
    """
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    files = [object_summary.key for object_summary in s3_bucket.objects.filter(Prefix=path)]
    return files


def csv_gen(bucket_name, files, skip=3):
    """
    read 3 files into memory
    shuffle df
    """
    for x in range(0, len(files), skip):
        df = pd.concat([pd.read_pickle("s3://"+bucket_name + "/" + file_path) 
                            for file_path in files[x:x+skip]]).sample(frac=1).reset_index(drop=True)
        yield df


def df_to_torch(data_df, rel_idx):
    return (
        torch.from_numpy(np.stack(data_df.padded_texts.iloc[rel_idx], axis=0)),
        torch.from_numpy(np.stack(data_df.padded_speakers.iloc[rel_idx], axis=0)).long(),
        torch.from_numpy(np.stack(data_df.padded_times.iloc[rel_idx], axis=0)).long(),
        torch.from_numpy(np.asarray(data_df.total_duration.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.ntt_count.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.ntt_duration.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.overtalk_count.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.overtalk_duration.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.talk_duration.iloc[rel_idx])).float(),
        torch.from_numpy(np.asarray(data_df.tnps.iloc[rel_idx])).long(),
    )

