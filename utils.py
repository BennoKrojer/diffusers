import numpy as np

RETRIEVAL_TASKS = ['imagecode', 'flickr30k']

def evaluate_winoground(scores):
    c0_i0, c0_i1, c1_i0, c1_i1 = scores[0]
    text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
    img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
    group_score = 1 if text_score and img_score else 0
    return text_score, img_score, group_score

def evaluate_retrieval(args, scores, img_idx):
    img_idx = img_idx.cpu().numpy()
    scores = scores.cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = 0
    for i in range(scores.shape[0]):
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy += 1
    retrieval_accuracy /= scores.shape[0]
    return retrieval_accuracy

def evaluate_scores(args, scores, batch):
    if args.task == 'winoground':
        score = evaluate_winoground(scores)
    elif args.task in RETRIEVAL_TASKS:
        img_idx = batch[-1]
        score = evaluate_retrieval(args, scores, img_idx)
    else:
        raise NotImplementedError
    return score