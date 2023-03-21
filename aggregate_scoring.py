from utils import evaluate_scores
import csv
from datasets_loading import get_dataset
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np

def main(args):
    
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None, scoring_only=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    score_dir = f'./cache/{args.task}/{"img2img" if args.img2img else "txt2img"}{"_strength" if args.strength else ""}'
    csv_files = [f for f in os.listdir(score_dir) if f.endswith('.csv')]
    print(csv_files)

    aggregated_scores = []
    for f in csv_files:
        with open(os.path.join(score_dir, f), 'r') as f:
            reader = csv.reader(f)
            scores = list(reader)
        all_scores = [[float(s) for s in datapoint] for datapoint in scores]
        aggregated_scores.append(all_scores)

    aggregated_scores = np.array(aggregated_scores)
    if args.deep_aggregation:
        aggregated_scores = aggregated_scores.transpose(1, 2, 0)

        progressing_scores = []
        for sample_size in range(aggregated_scores.shape[-1]):
            sub_aggregated_scores = aggregated_scores[:, :, :sample_size+1]
            metrics = {'mean': [], 'max': [], 'min': []}
            for i, batch in enumerate(dataloader):
                if args.subset and i % 10 != 0:
                    continue
                index = i//10 if args.subset else i
                scores = sub_aggregated_scores[index]
                mean_scores = np.mean(scores, axis=1)
                max_scores = np.max(scores, axis=1)
                min_scores = np.min(scores, axis=1)

                mean_metric = evaluate_scores(args, [mean_scores], batch)
                max_metric = evaluate_scores(args, [max_scores], batch)
                min_metric = evaluate_scores(args, [min_scores], batch)
                
                metrics['mean'].append(mean_metric)
                metrics['max'].append(max_metric)
                metrics['min'].append(min_metric)

            for k, v in metrics.items():
                print(f'Aggregation method: {k}')
                if args.task == 'winoground':
                    text_score = sum([m[0] for m in v]) / len(v)
                    img_score = sum([m[1] for m in v]) / len(v)
                    group_score = sum([m[2] for m in v]) / len(v)
                    print(f'Text score: {text_score}')
                    print(f'Image score: {img_score}')
                    print(f'Group score: {group_score}')
                else:
                    accuracy = sum(v) / len(v)
                    print(f'Retrieval Accuracy: {accuracy}')
                    if k == 'mean':
                        progressing_scores.append(accuracy)
        # plot progressing scores accumulatively
        import matplotlib.pyplot as plt
        plt.plot(progressing_scores)
        plt.xlabel('Number of Aggregated Scores')
        plt.ylabel('Retrieval Accuracy')
        plt.savefig(f'./cache/{args.task}/{"img2img" if args.img2img else "txt2img"}{"_strength" if args.strength else ""}/progressing_scores.png')

    else:
        aggregated_metrics = []
        for i, all_scores in enumerate(aggregated_scores):
            metrics = []
            for j, batch in enumerate(dataloader):
                if args.subset and j % 10 != 0:
                    continue
                index = j//10 if args.subset else j
                scores = [all_scores[index]]
                metric = evaluate_scores(args, scores, batch)
                metrics.append(metric)
            aggregated_metrics.append(metrics)
        aggregated_metrics = np.array(aggregated_metrics)
        if args.task == 'winoground':
            text_scores = aggregated_metrics[:,:,0]
            img_scores = aggregated_metrics[:,:,1]
            group_scores = aggregated_metrics[:,:,2]

            print(f'Text score: {np.mean(text_scores)}')
            print(f'Image score: {np.mean(img_scores)}')
            print(f'Group score: {np.mean(group_scores)}')
        else:
            print(f'Retrieval Accuracy: {np.mean(aggregated_metrics)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--img2img', action='store_true')
    parser.add_argument('--strength', action='store_true')
    parser.add_argument('--deep_aggregation', action='store_true')
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()

    main(args)