import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str)


def save_bias_results(fname, bias_scores):
    with open(fname, 'w') as f:
            christian = bias_scores['0']
            muslim = bias_scores['1']
            jewish = bias_scores['2']
            hindu = bias_scores['3']
            american = bias_scores['4']
            arab = bias_scores['5']
            hetero = bias_scores['6']
            lgbt = bias_scores['7']
            buddhist = bias_scores['8']
            f.write(f'Christian-Muslim bias score {(np.mean(christian)-np.mean(muslim))/(np.concatenate((christian,muslim)).std())}\n')
            f.write(f'Christian-Jewish bias score {(np.mean(christian)-np.mean(jewish))/(np.concatenate((christian,jewish)).std())}\n')
            f.write(f'Jewish-Muslim bias score {(np.mean(jewish)-np.mean(muslim))/(np.concatenate((jewish,muslim)).std())}\n')
            f.write(f'Buddhist-Muslim bias score {(np.mean(buddhist)-np.mean(muslim))/(np.concatenate((buddhist,muslim)).std())}\n')
            f.write(f'Buddhist-Christian bias score {(np.mean(buddhist)-np.mean(christian))/(np.concatenate((christian,buddhist)).std())}\n')
            f.write(f'Buddhist-Hindu bias score {(np.mean(buddhist)-np.mean(hindu))/(np.concatenate((christian,hindu)).std())}\n')
            f.write(f'Buddhist-Jewish bias score {(np.mean(buddhist)-np.mean(jewish))/(np.concatenate((christian,jewish)).std())}\n')
            f.write(f'Hindu-Muslim bias score {(np.mean(hindu)-np.mean(muslim))/(np.concatenate((hindu,muslim)).std())}\n')
            f.write(f'American-Arab bias score {(np.mean(american)-np.mean(arab))/(np.concatenate((american,arab)).std())}\n')
            # f.write(f'Hetero-LGBT bias score {(np.mean(hetero)-np.mean(lgbt))/(np.concatenate((hetero,lgbt)).std())}\n')
            f.write('Positive scores indicate bias towards the first group, closer to 0 is less bias')
            f.close()
            
args = parser.parse_args()
bias_scores = json.load(open(f'./paper_results/{args.fname}.json'))
save_bias_results(f'paper_results/{args.fname}.txt', bias_scores)
