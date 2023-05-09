import pandas as pd 
import plotly.express as px

data = [{'model':'Diffusion ITM 2.1','task':'Flickr Image','score':55.3},
        {'model':'Diffusion ITM 1.5','task':'Flickr Image','score':0},
        {'model':'Diffusion ITM 2.1','task':'ARO VG Attr.','score':59.2},
        {'model':'Diffusion ITM 1.5','task':'ARO VG Attr.','score':0},
        {'model':'Diffusion ITM 2.1','task':'ARO VG Rel.','score':49.8},
        {'model':'Diffusion ITM 1.5','task':'ARO VG Rel.','score':0},
        {'model':'Diffusion ITM 2.1','task':'COCO Order','score':24.8},
        {'model':'Diffusion ITM 1.5','task':'COCO Order','score':0},
        {'model':'Diffusion ITM 2.1','task':'Flickr Order','score':31.6},
        {'model':'Diffusion ITM 1.5','task':'Flickr Order','score':0},
        {'model':'Diffusion ITM 2.1','task':'CLEVR','score':65.7},
        {'model':'Diffusion ITM 1.5','task':'CLEVR','score':0},
        {'model':'Diffusion ITM 2.1','task':'Pets','score':62.5},
        {'model':'Diffusion ITM 1.5','task':'Pets','score':0},
        {'model':'Diffusion ITM 2.1','task':'Flickr Image','score':46.1},
        {'model':'Diffusion ITM 1.5','task':'Flickr Image','score':0},
        {'model':'Diffusion ITM 2.1','task':'SVO Verb','score':71.2},
        {'model':'Diffusion ITM 1.5','task':'SVO Verb','score':10},
        {'model':'Diffusion ITM 2.1','task':'SVO Subj','score':74.1},
        {'model':'Diffusion ITM 1.5','task':'SVO Subj','score':0},
        {'model':'Diffusion ITM 2.1','task':'SVO Obj','score':79.4},
        {'model':'Diffusion ITM 1.5','task':'SVO Obj','score':0},
        {'model':'Diffusion ITM 2.1','task':'ImageCoDe static','score':30.1},
        {'model':'Diffusion ITM 1.5','task':'ImageCoDe static','score':0},
        {'model':'Diffusion ITM 2.1','task':'ImageCoDe video','score':15.7},
        {'model':'Diffusion ITM 1.5','task':'ImageCoDe video','score':0},
        {'model':'Diffusion ITM 2.1','task':'Winoground group','score':7.5},
        {'model':'Diffusion ITM 1.5','task':'Winoground group','score':0},
        {'model':'Diffusion ITM 2.1','task':'Winoground text','score':32.3},
        {'model':'Diffusion ITM 1.5','task':'Winoground text','score':0},
        {'model':'Diffusion ITM 2.1','task':'Winoground image','score':9.0},
        {'model':'Diffusion ITM 1.5','task':'Winoground image','score':0},
        ]
# {'Flickr':{'Diffusion ITM 2.1':61.5,'Diffusion ITM 1.5':0},
#         'SVO_verb':{'Diffusion ITM 2.1':77.3,'Diffusion ITM 1.5':0},
#         'SVO_subj':{'Diffusion ITM 2.1':80.5,'Diffusion ITM 1.5':0},
#         'SVO_obj':{'Diffusion ITM 2.1':86.2,'Diffusion ITM 1.5':0},
#         'ImageCoDe_static':{'Diffusion ITM 2.1':42.5,'Diffusion ITM 1.5':0},
#         'ImageCoDe_video':{'Diffusion ITM 2.1':21.1 ,'Diffusion ITM 1.5':0},
#         'Winoground_group':{'Diffusion ITM 2.1':7.7 ,'Diffusion ITM 1.5':0},
#         'Winoground_text':{'Diffusion ITM 2.1':37.5 ,'Diffusion ITM 1.5':0},
#         'Winoground_image':{'Diffusion ITM 2.1':10.2,'Diffusion ITM 1.5':0}
#         }
df = pd.DataFrame(data)

import plotly.express as px
fig = px.bar(df, x="task", y="score", color="model",
              barmode='group', 
              labels={
                     "task": "Task",
                     "score": "Score",
                     "model": "Model"
                 },
             height=400)
# save fig
fig.write_image("1.5vs2.1.png")