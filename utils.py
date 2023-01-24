
def winoground_score():
    c0_i0, c0_i1, c1_i0, c1_i1 = score(i,metric=METRIC,caption_path=caption_path,image_path=image_path, use_clip=True)
    text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
    img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
    group_score = 1 if text_score and img_score else 0
    return text_score, img_score, group_score
