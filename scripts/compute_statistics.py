import argparse
import io
import json
import tarfile
from PIL import Image
import numpy as np
from p_tqdm import p_map
import os
from tqdm import tqdm
from collections import defaultdict, Counter
import logging
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import cloudpickle as cpickle


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process tar files in a directory')
    parser.add_argument('--max_files', type=int, default=-1, help='Max number of tar files to process (integer)')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level (string)')
    parser.add_argument('--total_workers', type=int, required=True, help='Total number of workers (integer)')
    parser.add_argument('--path', type=str, default='/p/fastdata/mmlaion/ocr/RenderedText/', help='path')
    parser.add_argument('--logdir', type=str, default='./log/', help='output path for logs and results')
    parser.add_argument('--type', type=str, default="tar", help='tar or parquet')
    parser.add_argument('--save_dict', action='store_true', help='save result dict')
    return parser.parse_args()

def bbox2str(bbox):
    res = '[(%.2f, %.2f)'%(bbox[0][0], bbox[0][1])
    for coord in bbox[1:]:
        res += ', (%.2f, %.2f)'%(coord[0], coord[1])
    res += ']'
    return res

def ocr_caption(json_content, ocr_key):
    if 'caption' not in json_content:
        ocr_caption = '<ocr>\n'
    else:
        ocr_caption = json_content['caption'] + '\n<ocr>\n'
    for bbox, text in zip(json_content[ocr_key]['bb_relative'], json_content[ocr_key]['text']):
        ocr_caption += bbox2str(bbox) + ': ' + text + '\n' 
    ocr_caption += '</ocr>' 
    return ocr_caption

def gpt4_process_text(text, remove_numbers=False):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    if remove_numbers:
        text = ''.join(ch for ch in text if ch.isalpha() or ch.isspace())
    else:
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
    # Tokenize
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)
            
def work_parquet(parquet_name):
    # do the same thing as work_tar but for parquet files
    statistics = defaultdict(list)
    # open parquet file
    df = pd.read_parquet(parquet_name)
    if 'ocr_annotation' in df.columns:
        ocr_key = 'ocr_annotation'
    elif 'ocr_result' in df.columns:
        ocr_key = 'ocr_result'
    else:
        ocr_key = None

    # iterate over rows
    for idx, row in tqdm(df.iterrows()):
        statistics['height'] += [row['height']]
        statistics['width'] += [row['width']]
        # check if ocr_annotation is present
        if ocr_key is not None:
            statistics['num_ocr_lines'] += [len(row[ocr_key]['text'])]
            statistics['ocr_lines'] += [row[ocr_key]['text'].tolist()]
            joined = ' '.join(statistics['ocr_lines'][-1])
            num_words = len(joined.split(' '))
            num_chars = len(joined) - statistics['num_ocr_lines'][-1] + 1
            statistics['num_ocr_words'] += [num_words]
            statistics['num_ocr_chars'] += [num_chars]

            statistics['ocr_caption'] += [ocr_caption(row, ocr_key)]
            statistics['len_ocr_caption'] += [len(statistics['ocr_caption'][-1])]
        # check if caption is present
        if 'caption' in row:
            statistics['len_caption'] += [len(row['caption'])]
    return statistics

def work_tar(tar_name):
    statistics = defaultdict(list)
    with tarfile.open(tar_name, mode='r') as input_tar:
        
        images = {}
        json_files = {}
        
        for tarinfo in input_tar:
            file_data = input_tar.extractfile(tarinfo)
            if file_data:
                if tarinfo.name.lower().endswith('.json'):
                    json_files[tarinfo.name] = json.load(file_data)
                elif tarinfo.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images[tarinfo.name] = file_data.read()
        
        
        for idx, (image_name, image_data) in tqdm(enumerate(images.items())):
            img_ending = image_name.split('.')[-1]
            json_name = image_name.replace(img_ending, 'json')
            logging.debug('Processing image: ' + image_name)
            logging.debug('Processing json: ' + json_name)
            if json_name in json_files:
                json_content = json_files[json_name]
                if 'ocr_annotation' in json_content:
                    ocr_key = 'ocr_annotation'
                elif 'ocr_result' in json_content:
                    ocr_key = 'ocr_result'
                else:
                    ocr_key = None

                try:
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    width, height = image.size
                    assert width == json_content['width'] and height == json_content['height'], 'json image dimensions missmatch with the loaded image'
                    statistics['height'] += [json_content['height']]
                    statistics['width'] += [json_content['width']]
                    if ocr_key is not None:
                        statistics['num_ocr_lines'] += [len(json_content[ocr_key]['text'])]
                        statistics['ocr_lines'] += [json_content[ocr_key]['text']]
                        joined = ' '.join(statistics['ocr_lines'][-1])
                        num_words = len(joined.split(' '))
                        num_chars = len(joined) - statistics['num_ocr_lines'][-1] + 1
                        statistics['num_ocr_words'] += [num_words]
                        statistics['num_ocr_chars'] += [num_chars]

                        statistics['ocr_caption'] += [ocr_caption(json_content, ocr_key)]
                        statistics['len_ocr_caption'] += [len(statistics['ocr_caption'][-1])]
                    if 'caption' in json_content:
                        statistics['len_caption'] += [len(json_content['caption'])]
                except Exception as e:
                    logging.error(f"Error processing image: {image_name} with error: {e}")
                
        return statistics

if __name__ == '__main__':
    args = parse_arguments()
    # create logdir if it does not exist already
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(args.logdir,'log.txt'))
    logging.info('Computing statistics over %s'%args.path)
    
    # create a list of tar files contained in args.path
    tar_files = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if args.max_files > 0 and len(tar_files) >= args.max_files:
                break
            if file.endswith('.'+args.type):
                tar_files += [os.path.join(root, file)]
            

    logging.info(f"Processing tar files: {len(tar_files)}")

    n = len(tar_files)
    if args.type == 'tar':
        results = p_map(work_tar, tar_files, num_cpus=args.total_workers)
    elif args.type == 'parquet':
        results = p_map(work_parquet, tar_files, num_cpus=args.total_workers)
    
    aggregate_result = {'total_number_of_imgs':0}
    all_results = defaultdict(list)
    for result in results:
        for key, value in result.items():
            all_results[key] += value
        all_results['num_imgs_per_tar'] += [len(result['height'])]
        aggregate_result['total_number_of_imgs'] += len(result['height'])
    logging.info(f"Total number of images: {aggregate_result['total_number_of_imgs']}")
        

    for key in all_results.keys():
        vals = all_results[key]
        if not (isinstance(vals[0], int) or isinstance(vals[0], float)):
            continue
        fig = plt.figure()
        plt.boxplot(all_results[key], showfliers=False)
        plt.title(f"Boxplot for {key}")
        plt.savefig(os.path.join(args.logdir, f'boxplot_{key}.png'))
        plt.close(fig)
        logging.info(f"Mean of {key}: {np.mean(all_results[key])}")
        logging.info(f"Std of {key}: {np.std(all_results[key])}")
        logging.info(f"Min of {key}: {np.min(all_results[key])}")
        logging.info(f"Max of {key}: {np.max(all_results[key])}")
        logging.info(f"Median of {key}: {np.median(all_results[key])}")
        logging.info(f"25th percentile of {key}: {np.quantile(all_results[key], 0.25)}")
        logging.info(f"75th percentile of {key}: {np.quantile(all_results[key], 0.75)}")
        logging.info(f"Interquartile range of {key}: {np.quantile(all_results[key], 0.75) - np.quantile(all_results[key], 0.25)}")
        logging.info(f"5th percentile of {key}: {np.quantile(all_results[key], 0.05)}")
        logging.info(f"95th percentile of {key}: {np.quantile(all_results[key], 0.95)}")
        aggregate_result[f"mean_{key}"] = float(np.mean(all_results[key]))
        aggregate_result[f"std_{key}"] = float(np.std(all_results[key]))
        aggregate_result[f"min_{key}"] = float(np.min(all_results[key]))
        aggregate_result[f"max_{key}"] = float(np.max(all_results[key]))
        aggregate_result[f"median_{key}"] = float(np.median(all_results[key]))
        aggregate_result[f"q1_{key}"] = float(np.quantile(all_results[key], 0.25))
        aggregate_result[f"q3_{key}"] = float(np.quantile(all_results[key], 0.75))
        aggregate_result[f"iqr_{key}"] = float(aggregate_result[f"q3_{key}"] - aggregate_result[f"q1_{key}"])
        aggregate_result[f"q05_{key}"] = float(np.quantile(all_results[key], 0.05))
        aggregate_result[f"q95_{key}"] = float(np.quantile(all_results[key], 0.95))

    # make word cloud aggregating the words in  all_results['ocr_lines']
    #print(all_results['ocr_lines'])
    all_ocr_lines = []
    for ocr_lines in all_results['ocr_lines']:
        all_ocr_lines += ocr_lines
    all_strings = ' '.join(all_ocr_lines)
    all_words = all_strings.split(' ')
    #remove leading and trailing spaces
    all_words = [word.strip() for word in all_words]

    # default wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(args.logdir, 'wordcloud_default.png'))
    plt.close()
    words_in_wordcloud = list(wordcloud.words_.keys())
    all_results['words_in_wordcloud'] = words_in_wordcloud
    all_results['words_in_wordcloud_freq'] = list(wordcloud.words_.values())
    logging.info(words_in_wordcloud)

    # ngram wordclouds
    def create_and_store_ngramwordcloud(n, words, fname, all_results, args):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        ngram_counts = Counter(ngrams)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(os.path.join(args.logdir, fname))
        plt.close()
        ngrams_in_wordcloud = list(wordcloud.words_.keys())
        logging.info(ngrams_in_wordcloud)
        all_results[f'{n}grams_in_'+fname] = ngrams_in_wordcloud
        all_results[f'{n}grams_freq_in_'+fname] = list(wordcloud.words_.values())
    
    all_words_and_numbers_processed = gpt4_process_text(' '.join(all_words), False).split(' ')
    all_words_processed = gpt4_process_text(' '.join(all_words), True).split(' ')

    create_and_store_ngramwordcloud(1, all_words_processed, 'wordcloud_1grams_processed.png', all_results, args)
    create_and_store_ngramwordcloud(2, all_words_processed, 'wordcloud_2grams_processed.png', all_results, args)
    create_and_store_ngramwordcloud(3, all_words_processed, 'wordcloud_3grams_processed.png', all_results, args)
    create_and_store_ngramwordcloud(1, all_words_and_numbers_processed, 'wordcloud_1grams_and_numbers_processed.png', all_results, args)
    create_and_store_ngramwordcloud(2, all_words_and_numbers_processed, 'wordcloud_2grams_and_numbers_processed.png', all_results, args)
    create_and_store_ngramwordcloud(3, all_words_and_numbers_processed, 'wordcloud_3grams_and_numbers_processed.png', all_results, args)

    with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
        json.dump(aggregate_result, f)
        
    if args.save_dict:
        with open(os.path.join(args.logdir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f)
    