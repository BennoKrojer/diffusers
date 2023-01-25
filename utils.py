RETRIEVAL_TASKS = ['imagecode', 'flickr30k']

def evaluate_winoground(scores):
    c0_i0, c0_i1, c1_i0, c1_i1 = scores
    text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
    img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
    group_score = 1 if text_score and img_score else 0
    return text_score, img_score, group_score

def evaluate_retrieval(args, scores, img_idx):
    img_idx = img_idx.cpu().numpy()
    scores = scores.cpu().numpy()
    retrieval_accuracy = 0
    for i in range(scores.shape[0]):
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy += 1
    retrieval_accuracy /= scores.shape[0]
    return retrieval_accuracy

def evaluate_scores(args, scores, batch):
    if args.task == 'winoground':
        text_score, img_score, group_score = evaluate_winoground(scores)
        print(f'Text score: {text_score}')
        print(f'Image score: {img_score}')
        print(f'Group score: {group_score}')
    elif args.task in RETRIEVAL_TASKS:
        img_idx = batch[-1]
        retrieval_accuracy = evaluate_retrieval(args, scores, img_idx)
        print(f'Retrieval accuracy: {retrieval_accuracy}')