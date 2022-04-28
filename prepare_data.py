import pandas as pd
import joblib
import sys
import csv
csv.field_size_limit(sys.maxsize)
from tqdm import tqdm
import boto3
import pickle
import tempfile
from io import StringIO, BytesIO

from sklearn.model_selection import train_test_split
from utils.preprocessors import HANPreprocessor


def save_to_s3(data, s3_client, bucket_name, key, file_type="joblib"):
        if file_type == "joblib":
            with tempfile.TemporaryFile() as fp:
                joblib.dump(data, fp)
                fp.seek(0)
                s3_client.upload_fileobj(fp, bucket_name, key, ExtraArgs={'ACL':'bucket-owner-full-control'})
        elif file_type == "csv":
            s3_resource = boto3.resource('s3')
            object = s3_resource.Object(bucket_name, key)
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False, sep="\t", header=True)
            object.put(Body=csv_buffer.getvalue(), ACL='bucket-owner-full-control')
        if file_type == "pkl":
            with tempfile.TemporaryFile() as fp:
                pickle.dump(data, fp)
                fp.seek(0)
                s3_client.upload_fileobj(fp, bucket_name, key, ExtraArgs={'ACL':'bucket-owner-full-control'})
                
                
def get_tokenizer_from_s3(bucket_name, key):
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket_name).download_fileobj(key, data)
        data.seek(0)    # move back to the beginning after writing
        tokenizer = joblib.load(data)
        return tokenizer
        
             
def init_tokenizer(sample_df, s3_client, bucket_name, name):
    text_tok = HANPreprocessor(q=0.7)
    text_tok.tokenize(sample_df['texts'].values, sample_df['participants'].values, sample_df['times'].values)
    save_to_s3(text_tok, s3_client, bucket_name, name)
    return text_tok


def transform(df, text_tok):
    df['padded_texts'], df['padded_speakers'], df['padded_times'] = text_tok.transform(df['texts'].values, 
                                                                                      df['participants'].values, 
                                                                                      df['times'].values) 
    return df


def extract_characteristics(row):
    total_duration = int(row.transcript[-1]['end_offset_millis'])
    ntt_count = int(row.characteristics['non_talk_time']['count'])
    ntt_duration = 1.0*row.characteristics['non_talk_time']['duration_millis']/total_duration if ntt_count else 0.0
    overtalk_count = int(row.characteristics['overtalk']['count'])
    overtalk_duration = 1.0*row.characteristics['overtalk']['duration_millis']/total_duration if overtalk_count else 0.0
    talk_duration = 1.0*row.characteristics['talk_time']['duration_millis']/total_duration if 'duration_millis' in row.characteristics['talk_time'] else 0
    return pd.Series([total_duration, ntt_count, ntt_duration, overtalk_count, overtalk_duration, talk_duration])


def extract_transcript(row):
    row.transcript = [x for x in row.transcript if x['participant'] != "system"]
    texts = [[x['content'].replace('\t', ' ').replace('\n', ' ').replace('\r', '').replace('    ', ' ')] for x in row.transcript]
    times = [[x['begin_offset_millis'], x['end_offset_millis']] for x in row.transcript]
    participants = [[x['participant']] for x in row.transcript]
    return pd.Series([texts, times, participants])


def clean_calls(s3_client, bucket_name, raw_prefix, filtered_prefix):
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    files = [object_summary.key for object_summary in s3_bucket.objects.filter(Prefix=raw_prefix) if
             object_summary.key.endswith('json')]
    ds = pd.DataFrame()
    total_before = 0
    total_after = 0
    total_reduced = 0
    for file in tqdm(files):
        df = pd.read_json('s3://{}/{}'.format(bucket_name, file), lines=True)
        rows_before = df.shape[0]
        total_before += rows_before
        df = df[df.transcript.notnull()]
        df = df[df.transcript.apply(len) >= 5]
        df[["total_duration", "ntt_count", "ntt_duration", "overtalk_count", "overtalk_duration", "talk_duration"]] = df.apply(extract_characteristics, axis=1)
        df[["texts", "times", "participants"]] = df.apply(extract_transcript, axis=1)
        df['original_tnps'] = df.tnps.copy()
        df.loc[df['tnps'] < 7, 'tnps'] = 0
        df.loc[df['tnps'] > 8, 'tnps'] = 1
        df.loc[df['tnps'] > 1, 'tnps'] = 2
        columns = ["texts", "times", "participants", "total_duration", "ntt_count", "ntt_duration", 
                   "overtalk_count", "overtalk_duration", "talk_duration", "original_tnps", "tnps"]
        df = df[columns]
        rows_after = df.shape[0]
        total_after += rows_after
        reduced = rows_before - rows_after
        total_reduced += reduced
        train_df, test_df = train_test_split(df, train_size=0.8)
        train_df, val_df = train_test_split(train_df, train_size=0.8)
        file_name = file.split("/")[-1].split('.')[0]+'.pkl'
        train_path = '{}train/{}'.format(filtered_prefix, file_name)
        test_path = '{}test/{}'.format(filtered_prefix, file_name)
        val_path = '{}val/{}'.format(filtered_prefix, file_name)
        save_to_s3(train_df, s3_client, bucket_name, train_path, file_type="pkl")
        save_to_s3(test_df, s3_client, bucket_name, test_path, file_type="pkl")
        save_to_s3(val_df, s3_client, bucket_name, val_path, file_type="pkl")
        ds = pd.concat([ds, train_df.sample(frac=0.2)])
    return ds
 

def tokenize_calls(s3_client, bucket_name, filtered_prefix, tokenized_prefix, tokenizer):
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    for p in ['train', 'test', 'val']:
        out_path = '{}{}/'.format(tokenized_prefix, p)
        prefix = filtered_prefix + p
        files = [object_summary.key for object_summary in s3_bucket.objects.filter(Prefix=prefix) if
                 object_summary.key.endswith('pkl')]
        for file in tqdm(files):
            df = pd.read_pickle('s3://{}/{}'.format(bucket_name, file))  
            df = transform(df, s3_client, bucket_name, file, tokenizer)
            file_name = file.split("/")[-1].split('.')[0]+'.pkl'
            file_p = '{}{}'.format(out_path, file_name)
            save_to_s3(df, s3_client, bucket_name, file_p, file_type="pkl")


